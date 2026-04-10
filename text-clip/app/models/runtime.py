import gc
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np
import openvino as ov

from .constants import (
    CLIP_EMBEDDING_DIMS,
    CONTEXT_LENGTH,
    LOG,
    MODEL_PATH,
    OV_CACHE_DIR,
    TEXT_CLIP_DEVICE,
)
from .qa_clip_tokenizer import create_full_tokenizer

_TOKENIZER = create_full_tokenizer()
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


class TextClipRuntime:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.model_base_path = Path(MODEL_PATH)
        self.qa_clip_path = self.model_base_path / "qa-clip" / "openvino"
        self.ov_cache_dir = Path(OV_CACHE_DIR)
        self.ov_cache_dir.mkdir(parents=True, exist_ok=True)
        self.core = ov.Core()
        self._compiled_model: Optional[ov.CompiledModel] = None
        self._infer_request: Optional[ov.InferRequest] = None
        self._configure_openvino_cache()

    def _configure_openvino_cache(self) -> None:
        try:
            self.core.set_property({"CACHE_DIR": str(self.ov_cache_dir)})
        except Exception as exc:
            LOG.warning("Failed to set Text-CLIP OpenVINO cache dir: %s", exc)

    def load(self) -> None:
        with self._lock:
            if self._compiled_model is not None and self._infer_request is not None:
                return

            text_model_path = self.qa_clip_path / "openvino_text_fp16.xml"
            if not text_model_path.exists():
                raise FileNotFoundError(f"Missing text model: {text_model_path}")

            compiled_model = self.core.compile_model(
                str(text_model_path),
                TEXT_CLIP_DEVICE,
                {"PERFORMANCE_HINT": "LATENCY"},
            )
            output_dim = compiled_model.outputs[0].get_partial_shape()[1].get_length()
            if output_dim != CLIP_EMBEDDING_DIMS:
                raise RuntimeError(
                    f"Text embedding dims mismatch: expected={CLIP_EMBEDDING_DIMS}, got={output_dim}"
                )

            self._compiled_model = compiled_model
            self._infer_request = compiled_model.create_infer_request()
            LOG.info("Text-CLIP model loaded on %s.", TEXT_CLIP_DEVICE)

    def get_text_embedding(self, text: str) -> List[float]:
        with self._lock:
            self.load()
            if self._compiled_model is None or self._infer_request is None:
                raise RuntimeError("Text-CLIP model is not loaded.")

            input_ids = _tokenize_for_clip([text], context_length=CONTEXT_LENGTH)
            attention_mask = np.array(input_ids != _PAD_TOKEN_ID, dtype=np.int64)

            self._infer_request.set_input_tensor(
                0,
                ov.Tensor(np.ascontiguousarray(input_ids), shared_memory=True),
            )
            self._infer_request.set_input_tensor(
                1,
                ov.Tensor(np.ascontiguousarray(attention_mask), shared_memory=True),
            )
            self._infer_request.infer()
            embeddings = np.asarray(self._infer_request.get_output_tensor(0).data)

            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            if embeddings.shape[-1] != CLIP_EMBEDDING_DIMS:
                raise RuntimeError(
                    "Invalid text embedding dims: "
                    f"expected={CLIP_EMBEDDING_DIMS}, got={embeddings.shape[-1]}"
                )
            return embeddings.astype(np.float32, copy=False)[0].tolist()

    def release(self) -> None:
        with self._lock:
            self._infer_request = None
            self._compiled_model = None
        gc.collect()
