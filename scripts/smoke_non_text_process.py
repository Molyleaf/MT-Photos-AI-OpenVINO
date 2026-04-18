import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from non_text_process import NonTextProcessManager


_ORDERED_OPERATIONS = ("clip_img", "ocr", "represent")
_ORDERED_FAMILIES = {
    "clip_img": "vision",
    "ocr": "ocr",
    "represent": "face",
}


def _build_operation_result(operation: str, pid: int):
    if operation == "clip_img":
        return [float(pid)]
    if operation == "ocr":
        return {
            "texts": [str(pid)],
            "scores": ["1.0"],
            "boxes": [{"x": "0", "y": "0", "width": "1", "height": "1"}],
        }
    if operation == "represent":
        return [
            {
                "embedding": [float(pid)],
                "facial_area": {"x": 0, "y": 0, "w": 1, "h": 1},
                "face_confidence": 1.0,
            }
        ]
    raise RuntimeError(f"unsupported {operation}")


async def _invoke_operation(manager: NonTextProcessManager, operation: str, image: np.ndarray):
    if operation == "clip_img":
        return await manager.get_image_embedding_async(image)
    if operation == "ocr":
        return await manager.get_ocr_results_async(image)
    return await manager.get_face_representation_async(image)


def _extract_operation_pid(operation: str, payload) -> int:
    if operation == "clip_img":
        return int(payload[0])
    if operation == "ocr":
        return int(payload.texts[0])
    return int(payload[0].embedding[0])


def _fake_non_text_worker(request_queue, response_queue) -> None:
    pid = os.getpid()
    response_queue.put({"kind": "ready", "pid": pid})
    operation_delay_seconds = {
        "clip_img": 0.04,
        "ocr": 0.06,
        "represent": 0.18,
    }
    while True:
        message = request_queue.get()
        kind = str(message.get("kind", ""))
        if kind == "shutdown":
            request_queue.task_done()
            break
        request_id = str(message["request_id"])
        operation = str(message["operation"])
        time.sleep(operation_delay_seconds.get(operation, 0.02))
        try:
            result = _build_operation_result(operation, pid)
        except RuntimeError as exc:
            response_queue.put(
                {"kind": "error", "request_id": request_id, "error": str(exc)}
            )
            request_queue.task_done()
            continue
        response_queue.put(
            {
                "kind": "result",
                "request_id": request_id,
                "operation": operation,
                "result": result,
            }
        )
        request_queue.task_done()
    response_queue.put({"kind": "stopped", "pid": pid})


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Non-text worker mixed-order smoke and micro-benchmark.",
    )
    parser.add_argument(
        "--mixed-burst",
        type=int,
        default=18,
        help="total mixed requests for the pressure round",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=6,
        help="max concurrent caller coroutines for the pressure round",
    )
    return parser


def _percentile(values_ms: list[float], percentile: float) -> float:
    if not values_ms:
        return 0.0
    if len(values_ms) == 1:
        return values_ms[0]
    ordered = sorted(values_ms)
    rank = max(0.0, min(1.0, percentile / 100.0)) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(len(ordered) - 1, lower + 1)
    weight = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


async def _run_sequential_sanity(manager: NonTextProcessManager, image: np.ndarray) -> dict[str, int | str]:
    clip_result = await manager.get_image_embedding_async(image)
    ocr_result = await manager.get_ocr_results_async(image)
    await asyncio.to_thread(manager.release_models_for_restart)
    face_result = await manager.get_face_representation_async(image)
    summary = {
        "clip_pid": int(clip_result[0]),
        "ocr_pid": int(ocr_result.texts[0]),
        "face_pid": int(face_result[0].embedding[0]),
        "loaded_family": manager.get_loaded_runtime_family() or "none",
    }
    if summary["clip_pid"] == summary["ocr_pid"]:
        raise RuntimeError("Family switch did not restart the non-text worker.")
    if summary["loaded_family"] != "face":
        raise RuntimeError(f"Unexpected loaded family after represent: {summary['loaded_family']}")
    return summary


async def _run_out_of_order_round(manager: NonTextProcessManager, image: np.ndarray) -> dict[str, int | str]:
    face_task = asyncio.create_task(_invoke_operation(manager, "represent", image))

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if manager.get_loaded_runtime_family() == "face":
            break
        await asyncio.sleep(0.02)
    else:
        raise RuntimeError("Face request did not acquire the worker in time.")

    clip_task = asyncio.create_task(_invoke_operation(manager, "clip_img", image))
    first_done, _pending = await asyncio.wait(
        {face_task, clip_task},
        timeout=6.0,
        return_when=asyncio.FIRST_COMPLETED,
    )
    if face_task not in first_done:
        raise RuntimeError("Out-of-order round violated first-request-first-complete ordering.")

    face_result, clip_result = await asyncio.wait_for(
        asyncio.gather(face_task, clip_task),
        timeout=6.0,
    )
    face_pid = _extract_operation_pid("represent", face_result)
    clip_pid = _extract_operation_pid("clip_img", clip_result)
    if face_pid == clip_pid:
        raise RuntimeError("Out-of-order round did not recycle the worker between face and clip.")
    if manager.get_loaded_runtime_family() != "vision":
        raise RuntimeError(
            f"Unexpected loaded family after mixed round: {manager.get_loaded_runtime_family()}"
        )
    return {
        "face_pid": face_pid,
        "clip_pid": clip_pid,
        "loaded_family": manager.get_loaded_runtime_family() or "none",
    }


async def _run_pairwise_rotation_rounds(
    manager: NonTextProcessManager,
    image: np.ndarray,
) -> list[dict[str, object]]:
    rounds = []
    for first_operation in _ORDERED_OPERATIONS:
        for second_operation in _ORDERED_OPERATIONS:
            if first_operation == second_operation:
                continue

            await asyncio.to_thread(manager.release_models_for_restart)
            first_task = asyncio.create_task(_invoke_operation(manager, first_operation, image))

            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if manager.get_loaded_runtime_family() == _ORDERED_FAMILIES[first_operation]:
                    break
                await asyncio.sleep(0.02)
            else:
                raise RuntimeError(f"{first_operation} did not acquire the worker in time.")

            second_task = asyncio.create_task(_invoke_operation(manager, second_operation, image))
            first_completed, _pending = await asyncio.wait(
                {first_task, second_task},
                timeout=6.0,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if first_task not in first_completed:
                raise RuntimeError(
                    f"Rotation {first_operation}->{second_operation} let the second request complete first."
                )

            first_result = await asyncio.wait_for(first_task, timeout=6.0)
            second_result = await asyncio.wait_for(second_task, timeout=6.0)
            rounds.append(
                {
                    "first_operation": first_operation,
                    "second_operation": second_operation,
                    "first_pid": _extract_operation_pid(first_operation, first_result),
                    "second_pid": _extract_operation_pid(second_operation, second_result),
                    "loaded_family": manager.get_loaded_runtime_family() or "none",
                }
            )
    return rounds


async def _run_pressure_round(
    manager: NonTextProcessManager,
    image: np.ndarray,
    *,
    mixed_burst: int,
    concurrency: int,
) -> dict[str, object]:
    operations = ["represent", "clip_img", "ocr"]
    semaphore = asyncio.Semaphore(max(1, concurrency))
    records: list[dict[str, float | int | str]] = []

    async def invoke(index: int) -> None:
        operation = operations[index % len(operations)]
        delay_seconds = 0.01 * (index % max(1, concurrency))
        await asyncio.sleep(delay_seconds)
        async with semaphore:
            started = time.perf_counter()
            payload = await _invoke_operation(manager, operation, image)
            pid = _extract_operation_pid(operation, payload)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            records.append(
                {
                    "operation": operation,
                    "latency_ms": elapsed_ms,
                    "pid": pid,
                }
            )

    started = time.perf_counter()
    await asyncio.wait_for(
        asyncio.gather(*(invoke(index) for index in range(max(1, mixed_burst)))),
        timeout=max(12.0, mixed_burst * 1.2),
    )
    total_elapsed_ms = (time.perf_counter() - started) * 1000.0

    latency_by_operation: dict[str, list[float]] = {}
    worker_pids = set()
    for record in records:
        operation = str(record["operation"])
        latency_by_operation.setdefault(operation, []).append(float(record["latency_ms"]))
        worker_pids.add(int(record["pid"]))

    operation_summary = {}
    for operation, latencies in latency_by_operation.items():
        operation_summary[operation] = {
            "count": len(latencies),
            "avg_ms": round(statistics.fmean(latencies), 2),
            "p95_ms": round(_percentile(latencies, 95.0), 2),
            "max_ms": round(max(latencies), 2),
        }

    return {
        "request_count": len(records),
        "concurrency": max(1, concurrency),
        "total_elapsed_ms": round(total_elapsed_ms, 2),
        "worker_restart_count": max(0, len(worker_pids) - 1),
        "operations": operation_summary,
    }


async def main() -> int:
    args = _build_argument_parser().parse_args()

    os.environ.setdefault("INFERENCE_QUEUE_MAX_SIZE", "4")
    os.environ.setdefault("INFERENCE_QUEUE_TIMEOUT", "3")
    os.environ.setdefault("INFERENCE_EXEC_TIMEOUT", "3")
    os.environ.setdefault("OCR_EXEC_TIMEOUT", "3")
    os.environ.setdefault("NON_TEXT_IDLE_RELEASE_SECONDS", "0")

    manager = NonTextProcessManager(worker_target=_fake_non_text_worker)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    try:
        summary = {
            "sequential_sanity": await _run_sequential_sanity(manager, image),
            "out_of_order_round": await _run_out_of_order_round(manager, image),
            "pairwise_rotations": await _run_pairwise_rotation_rounds(manager, image),
            "pressure_round": await _run_pressure_round(
                manager,
                image,
                mixed_burst=max(1, int(args.mixed_burst)),
                concurrency=max(1, int(args.concurrency)),
            ),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    finally:
        await asyncio.to_thread(manager.release_all_models)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
