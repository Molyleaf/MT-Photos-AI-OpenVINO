import asyncio
import json
import os
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .constants import (
    CLIP_EMBEDDING_DIMS,
    EXEC_TIMEOUT_SECONDS,
    TEXT_CLIP_BASE_URL,
)


class ClipTextMixin:
    _text_clip_base_url: str
    _text_clip_timeout_seconds: float
    _text_clip_api_key: str

    def _initialize_text_clip_client(self) -> None:
        base_url = str(os.environ.get("TEXT_CLIP_BASE_URL", TEXT_CLIP_BASE_URL)).strip()
        if not base_url:
            raise RuntimeError("TEXT_CLIP_BASE_URL is empty.")

        self._text_clip_base_url = base_url.rstrip("/")
        self._text_clip_timeout_seconds = float(max(1, EXEC_TIMEOUT_SECONDS))
        self._text_clip_api_key = str(
            os.environ.get("TEXT_CLIP_API_KEY", os.environ.get("API_AUTH_KEY", ""))
        ).strip()

    @staticmethod
    def _decode_json_response(payload: bytes) -> Dict[str, Any]:
        if not payload:
            raise RuntimeError("Text-CLIP service returned empty response.")
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError("Text-CLIP service returned invalid JSON.") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("Text-CLIP service returned invalid payload.")
        return decoded

    def _build_text_clip_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._text_clip_api_key and self._text_clip_api_key.lower() != "no-key":
            headers["api-key"] = self._text_clip_api_key
        return headers

    def _post_text_clip_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            url=f"{self._text_clip_base_url}{path}",
            data=body,
            headers=self._build_text_clip_headers(),
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._text_clip_timeout_seconds) as response:
                return self._decode_json_response(response.read())
        except HTTPError as exc:
            raw_payload = exc.read()
            detail = ""
            if raw_payload:
                try:
                    decoded = self._decode_json_response(raw_payload)
                except Exception:
                    detail = raw_payload.decode("utf-8", errors="replace").strip()
                else:
                    detail = str(
                        decoded.get("detail") or decoded.get("msg") or decoded.get("error") or ""
                    ).strip()
            reason = detail or str(exc.reason or f"HTTP {exc.code}")
            raise RuntimeError(
                f"Text-CLIP service request failed: HTTP {exc.code} {reason}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"Text-CLIP service is unavailable: {exc.reason}") from exc
        except OSError as exc:
            raise RuntimeError(f"Text-CLIP service request failed: {exc}") from exc

    def get_text_embedding(self, text: str) -> List[float]:
        response = self._post_text_clip_json("/clip/txt", {"text": text})
        message = str(response.get("msg", "") or "").strip()
        if message:
            raise RuntimeError(message)

        result = response.get("result")
        if not isinstance(result, list):
            raise RuntimeError("Text-CLIP service returned invalid payload.")

        try:
            embedding = [float(item) for item in result]
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Text-CLIP service returned invalid embedding values.") from exc

        if len(embedding) != CLIP_EMBEDDING_DIMS:
            raise RuntimeError(
                "Text-CLIP service returned invalid embedding dims: "
                f"expected={CLIP_EMBEDDING_DIMS}, got={len(embedding)}"
            )
        return embedding

    async def get_text_embedding_async(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.get_text_embedding, text)
