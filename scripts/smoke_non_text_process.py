import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from non_text_process import NonTextProcessManager


def _fake_non_text_worker(request_queue, response_queue) -> None:
    pid = os.getpid()
    response_queue.put({"kind": "ready", "pid": pid})
    while True:
        message = request_queue.get()
        kind = str(message.get("kind", ""))
        if kind == "shutdown":
            request_queue.task_done()
            break
        request_id = str(message["request_id"])
        operation = str(message["operation"])
        if operation == "clip_img":
            result = [float(pid)]
        elif operation == "ocr":
            result = {
                "texts": [str(pid)],
                "scores": ["1.0"],
                "boxes": [{"x": "0", "y": "0", "width": "1", "height": "1"}],
            }
        elif operation == "represent":
            result = [
                {
                    "embedding": [float(pid)],
                    "facial_area": {"x": 0, "y": 0, "w": 1, "h": 1},
                    "face_confidence": 1.0,
                }
            ]
        else:
            response_queue.put(
                {"kind": "error", "request_id": request_id, "error": f"unsupported {operation}"}
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


async def main() -> int:
    os.environ.setdefault("INFERENCE_QUEUE_MAX_SIZE", "4")
    os.environ.setdefault("INFERENCE_QUEUE_TIMEOUT", "3")
    os.environ.setdefault("INFERENCE_EXEC_TIMEOUT", "3")
    os.environ.setdefault("OCR_EXEC_TIMEOUT", "3")
    os.environ.setdefault("NON_TEXT_IDLE_RELEASE_SECONDS", "0")

    manager = NonTextProcessManager(worker_target=_fake_non_text_worker)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    try:
        clip_result = await manager.get_image_embedding_async(image)
        ocr_result = await manager.get_ocr_results_async(image)
        await asyncio.to_thread(manager.release_models_for_restart)
        face_result = await manager.get_face_representation_async(image)
        summary = {
            "clip_pid": int(clip_result[0]),
            "ocr_pid": int(ocr_result.texts[0]),
            "face_pid": int(face_result[0].embedding[0]),
            "loaded_family": manager.get_loaded_runtime_family(),
        }
        if summary["clip_pid"] == summary["ocr_pid"]:
            raise RuntimeError("Family switch did not restart the non-text worker.")
        if summary["loaded_family"] != "face":
            raise RuntimeError(f"Unexpected loaded family after represent: {summary['loaded_family']}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    finally:
        await asyncio.to_thread(manager.release_all_models)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
