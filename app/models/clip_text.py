import asyncio
import json
import socket
import socketserver
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import numpy as np
import openvino as ov

from .common import _TextClipRpcServer
from .constants import (
    CLIP_EMBEDDING_DIMS,
    CONTEXT_LENGTH,
    LOG,
    PROCESS_LOCK_POLL_SECONDS,
    QA_CLIP_CLIP_ROOT,
    TEXT_RPC_HOST,
)

if str(QA_CLIP_CLIP_ROOT) not in sys.path:
    sys.path.insert(0, str(QA_CLIP_CLIP_ROOT))

# noinspection PyUnresolvedReferences
from bert_tokenizer import FullTokenizer  # noqa: E402

_TOKENIZER = FullTokenizer()
_PAD_TOKEN_ID = int(_TOKENIZER.vocab["[PAD]"])
_CLS_TOKEN_ID = int(_TOKENIZER.vocab["[CLS]"])
_SEP_TOKEN_ID = int(_TOKENIZER.vocab["[SEP]"])


def _tokenize_for_clip(texts: List[str], context_length: int = CONTEXT_LENGTH) -> np.ndarray:
    if context_length < 2:
        raise ValueError("context_length must be >= 2")

    token_rows: List[List[int]] = []
    for text in texts:
        token_ids = _TOKENIZER.convert_tokens_to_ids(_TOKENIZER.tokenize(text))
        token_ids = token_ids[: context_length - 2]
        row = [_CLS_TOKEN_ID, *token_ids, _SEP_TOKEN_ID]
        token_rows.append(row[:context_length])

    result = np.zeros((len(token_rows), context_length), dtype=np.int64)
    for index, row in enumerate(token_rows):
        result[index, : len(row)] = np.asarray(row, dtype=np.int64)
    return result


class ClipTextMixin:
    _pid: int
    _execution_timeout_seconds: int
    _clip_inference_device: str
    qa_clip_path: Path
    _clip_remote_context: Any
    _clip_text_model: Optional[ov.CompiledModel]
    _clip_text_request: Optional[ov.InferRequest]
    _clip_text_input_names: Optional[Tuple[str, str]]
    _clip_text_host_input_cache: Dict[int, Tuple[ov.Tensor, Any, ov.Tensor, Any]]
    _clip_text_host_tensor_enabled: bool
    _clip_text_lock: Any
    _text_service_meta_path: Path
    _text_service_lock: Any
    _text_service_owner: bool
    _text_service_server: Optional[_TextClipRpcServer]
    _text_service_thread: Optional[threading.Thread]
    _text_service_port: Optional[int]

    if TYPE_CHECKING:
        def _compile_clip_model(
            self,
            model_or_path: Any,
            performance_hint: str,
        ) -> ov.CompiledModel: ...

    def _read_text_service_meta(self) -> Optional[Dict[str, Any]]:
        try:
            raw = json.loads(self._text_service_meta_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except Exception as exc:
            LOG.warning("Failed to read Text-CLIP service metadata: %s", exc)
            return None
        if not isinstance(raw, dict):
            return None
        return raw

    def _write_text_service_meta(self, port: int) -> None:
        payload = {"port": int(port), "pid": self._pid}
        self._text_service_meta_path.write_text(
            json.dumps(payload, ensure_ascii=True),
            encoding="utf-8",
        )

    def _probe_text_service(self, port: int, timeout_seconds: float = 0.5) -> bool:
        try:
            with socket.create_connection((TEXT_RPC_HOST, int(port)), timeout=timeout_seconds) as conn:
                request = {"op": "ping"}
                conn.sendall((json.dumps(request, ensure_ascii=True) + "\n").encode("utf-8"))
                conn.shutdown(socket.SHUT_WR)
                response_line = b""
                while not response_line.endswith(b"\n"):
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    response_line += chunk
        except OSError:
            return False

        if not response_line:
            return False
        try:
            response = json.loads(response_line.decode("utf-8"))
        except Exception:
            return False
        return response.get("ok") is True

    def _load_text_model_once_locked(self) -> None:
        if self._clip_text_model is not None and self._clip_text_request is not None:
            return
        self._load_clip_text_locked()

    def _load_text_model_once(self) -> None:
        with self._clip_text_lock:
            self._load_text_model_once_locked()

    def _start_text_service_locked(self) -> None:
        if self._text_service_server is not None and self._text_service_thread is not None:
            return

        self._load_text_model_once_locked()
        parent = self

        class _TextClipRequestHandler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                try:
                    request_line = self.rfile.readline()
                    if not request_line:
                        return
                    request = json.loads(request_line.decode("utf-8"))
                    op = str(request.get("op", "")).strip().lower()
                    if op == "ping":
                        response = {"ok": True}
                    elif op == "embed":
                        text = str(request.get("text", ""))
                        response = {"result": parent._infer_text_locally(text)}
                    else:
                        response = {"error": f"Unsupported op: {op or 'missing'}"}
                except Exception as exc:
                    response = {"error": str(exc)}
                self.wfile.write((json.dumps(response, ensure_ascii=True) + "\n").encode("utf-8"))

        handler_class = cast(type[socketserver.BaseRequestHandler], _TextClipRequestHandler)
        server = _TextClipRpcServer((TEXT_RPC_HOST, 0), handler_class)
        self._text_service_server = server
        self._text_service_port = int(server.server_address[1])
        self._write_text_service_meta(self._text_service_port)
        self._text_service_thread = threading.Thread(
            target=server.serve_forever,
            name="text-clip-rpc",
            daemon=True,
        )
        self._text_service_thread.start()
        LOG.info(
            "Text-CLIP RPC service started on %s:%s in pid=%s.",
            TEXT_RPC_HOST,
            self._text_service_port,
            self._pid,
        )

    def _ensure_text_service_ready(self, preload: bool) -> None:
        if self._text_service_owner and self._text_service_port is not None:
            if preload:
                self._load_text_model_once()
            return

        metadata = self._read_text_service_meta()
        if metadata is not None:
            candidate_port = int(metadata.get("port", 0) or 0)
            if candidate_port > 0 and self._probe_text_service(candidate_port):
                self._text_service_port = candidate_port
                return

        acquired = self._text_service_lock.acquire(timeout=0.0, blocking=False)
        if not acquired:
            deadline = time.monotonic() + max(3.0, float(self._execution_timeout_seconds))
            while time.monotonic() < deadline:
                metadata = self._read_text_service_meta()
                if metadata is not None:
                    candidate_port = int(metadata.get("port", 0) or 0)
                    if candidate_port > 0 and self._probe_text_service(candidate_port):
                        self._text_service_port = candidate_port
                        return
                time.sleep(PROCESS_LOCK_POLL_SECONDS)
            raise RuntimeError("Text-CLIP RPC service is unavailable.")

        self._text_service_owner = True
        try:
            metadata = self._read_text_service_meta()
            if metadata is not None:
                candidate_port = int(metadata.get("port", 0) or 0)
                if candidate_port > 0 and self._probe_text_service(candidate_port):
                    self._text_service_port = candidate_port
                    return
            self._start_text_service_locked()
        except Exception:
            self._text_service_owner = False
            self._text_service_lock.release()
            raise

    def _infer_text_locally(self, text: str) -> List[float]:
        with self._clip_text_lock:
            self._load_text_model_once_locked()
            return self._infer_clip_text_batch([text])[0]

    def _request_text_embedding_remote(self, text: str) -> List[float]:
        self._ensure_text_service_ready(preload=False)
        if self._text_service_owner:
            return self._infer_text_locally(text)
        if self._text_service_port is None:
            raise RuntimeError("Text-CLIP RPC service port is unavailable.")

        response_line = b""
        try:
            with socket.create_connection(
                (TEXT_RPC_HOST, self._text_service_port),
                timeout=max(1.0, float(self._execution_timeout_seconds)),
            ) as conn:
                request = {"op": "embed", "text": text}
                conn.sendall((json.dumps(request, ensure_ascii=True) + "\n").encode("utf-8"))
                conn.shutdown(socket.SHUT_WR)
                while not response_line.endswith(b"\n"):
                    chunk = conn.recv(16384)
                    if not chunk:
                        break
                    response_line += chunk
        except OSError:
            self._text_service_port = None
            self._ensure_text_service_ready(preload=False)
            if self._text_service_owner:
                return self._infer_text_locally(text)
            if self._text_service_port is None:
                raise RuntimeError("Text-CLIP RPC service port is unavailable.")
            response_line = b""
            with socket.create_connection(
                (TEXT_RPC_HOST, self._text_service_port),
                timeout=max(1.0, float(self._execution_timeout_seconds)),
            ) as conn:
                request = {"op": "embed", "text": text}
                conn.sendall((json.dumps(request, ensure_ascii=True) + "\n").encode("utf-8"))
                conn.shutdown(socket.SHUT_WR)
                while not response_line.endswith(b"\n"):
                    chunk = conn.recv(16384)
                    if not chunk:
                        break
                    response_line += chunk

        if not response_line:
            raise RuntimeError("Text-CLIP RPC service returned empty response.")
        response = json.loads(response_line.decode("utf-8"))
        if "error" in response:
            raise RuntimeError(str(response["error"]))
        result = response.get("result")
        if not isinstance(result, list):
            raise RuntimeError("Text-CLIP RPC service returned invalid payload.")
        return [float(item) for item in result]

    def _shutdown_text_service(self) -> None:
        if self._text_service_server is not None:
            self._text_service_server.shutdown()
            self._text_service_server.server_close()
            self._text_service_server = None
        if self._text_service_thread is not None:
            self._text_service_thread.join(timeout=2.0)
            self._text_service_thread = None
        if self._text_service_owner:
            self._text_service_lock.release()
            self._text_service_owner = False
        if self._text_service_meta_path.exists():
            metadata = self._read_text_service_meta()
            if metadata is not None and int(metadata.get("pid", -1)) == self._pid:
                try:
                    self._text_service_meta_path.unlink()
                except OSError:
                    pass

    def _get_text_host_tensors(
        self, batch_size: int
    ) -> Optional[Tuple[ov.Tensor, np.ndarray, ov.Tensor, np.ndarray]]:
        if (
            not self._clip_text_host_tensor_enabled
            or self._clip_remote_context is None
            or self._clip_text_model is None
        ):
            return None

        cached = self._clip_text_host_input_cache.get(batch_size)
        if cached is not None:
            return cached

        try:
            tensor_shape = ov.Shape([int(batch_size), int(CONTEXT_LENGTH)])
            input_0 = self._clip_text_model.inputs[0]
            input_1 = self._clip_text_model.inputs[1]
            input_tensor_0 = self._clip_remote_context.create_host_tensor(
                input_0.get_element_type(), tensor_shape
            )
            input_tensor_1 = self._clip_remote_context.create_host_tensor(
                input_1.get_element_type(), tensor_shape
            )
            input_view_0 = np.asarray(input_tensor_0.data)
            input_view_1 = np.asarray(input_tensor_1.data)
            entry = (input_tensor_0, input_view_0, input_tensor_1, input_view_1)
            self._clip_text_host_input_cache[batch_size] = entry
            return entry
        except Exception as exc:
            LOG.warning(
                "CLIP text host tensor allocation failed, fallback to shared numpy inputs: %s",
                exc,
            )
            self._clip_text_host_tensor_enabled = False
            self._clip_text_host_input_cache.clear()
            return None

    def _load_clip_text_locked(self) -> None:
        text_model_path = self.qa_clip_path / "openvino_text_fp16.xml"
        if not text_model_path.exists():
            raise FileNotFoundError(f"Missing text model: {text_model_path}")

        compiled_model = self._compile_clip_model(
            model_or_path=text_model_path,
            performance_hint="LATENCY",
        )
        output_dim = compiled_model.outputs[0].get_partial_shape()[1].get_length()
        if output_dim != CLIP_EMBEDDING_DIMS:
            raise RuntimeError(
                f"Text embedding dims mismatch: expected={CLIP_EMBEDDING_DIMS}, got={output_dim}"
            )

        self._clip_text_model = compiled_model
        self._clip_text_request = compiled_model.create_infer_request()
        self._clip_text_input_names = (
            self._clip_text_model.inputs[0].any_name,
            self._clip_text_model.inputs[1].any_name,
        )
        self._clip_text_host_input_cache.clear()
        self._clip_text_host_tensor_enabled = self._clip_remote_context is not None
        LOG.info("CLIP Text model loaded on %s.", self._clip_inference_device)

    def _infer_clip_text_batch(self, texts: List[str]) -> List[List[float]]:
        if not self._clip_text_model or not self._clip_text_request:
            raise RuntimeError("CLIP text model is not loaded.")
        if self._clip_text_input_names is None:
            raise RuntimeError("CLIP text input metadata is not initialized.")

        input_ids = _tokenize_for_clip(texts, context_length=CONTEXT_LENGTH)
        attention_mask = np.array(input_ids != _PAD_TOKEN_ID, dtype=np.int64)

        host_tensors = self._get_text_host_tensors(batch_size=input_ids.shape[0])
        if host_tensors is not None:
            input_tensor_0, input_view_0, input_tensor_1, input_view_1 = host_tensors
            np.copyto(input_view_0, input_ids, casting="no")
            np.copyto(input_view_1, attention_mask, casting="no")
            self._clip_text_request.set_input_tensor(0, input_tensor_0)
            self._clip_text_request.set_input_tensor(1, input_tensor_1)
            self._clip_text_request.infer()
            embeddings = np.asarray(self._clip_text_request.get_output_tensor(0).data)
        else:
            self._clip_text_request.set_input_tensor(
                0,
                ov.Tensor(np.ascontiguousarray(input_ids), shared_memory=True),
            )
            self._clip_text_request.set_input_tensor(
                1,
                ov.Tensor(np.ascontiguousarray(attention_mask), shared_memory=True),
            )
            self._clip_text_request.infer()
            embeddings = np.asarray(self._clip_text_request.get_output_tensor(0).data)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape[-1] != CLIP_EMBEDDING_DIMS:
            raise RuntimeError(
                f"Invalid text embedding dims: expected={CLIP_EMBEDDING_DIMS}, got={embeddings.shape[-1]}"
            )
        return embeddings.astype(np.float32, copy=False).tolist()

    def ensure_clip_text_model_loaded(self) -> None:
        self._ensure_text_service_ready(preload=True)

    async def ensure_clip_text_model_loaded_async(self) -> None:
        await asyncio.to_thread(self.ensure_clip_text_model_loaded)

    def get_text_embedding(self, text: str) -> List[float]:
        return self._request_text_embedding_remote(text)

    async def get_text_embedding_async(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.get_text_embedding, text)
