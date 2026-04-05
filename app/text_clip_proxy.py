import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from bootstrap import TextClipProxySettings


def resolve_text_clip_endpoint_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError(
            "TEXT_CLIP_SERVER_URL 必须是绝对 http(s) URL，例如 "
            "'http://127.0.0.1:8061' 或 'http://mt-photos-ai-text-clip:8061'。"
        )

    normalized_path = parsed.path.rstrip("/")
    if normalized_path.endswith("/clip/txt"):
        endpoint_path = normalized_path or "/clip/txt"
    elif normalized_path:
        endpoint_path = f"{normalized_path}/clip/txt"
    else:
        endpoint_path = "/clip/txt"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, endpoint_path, "", ""))


def decode_text_clip_response(raw_body: bytes, *, status_code: int) -> dict[str, Any]:
    response_text = raw_body.decode("utf-8", errors="replace").strip()
    if not response_text:
        if 200 <= status_code < 300:
            raise RuntimeError("独立 Text-CLIP 服务返回了空响应。")
        return {"result": [], "msg": f"独立 Text-CLIP 服务请求失败（HTTP {status_code}）。"}

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        if 200 <= status_code < 300:
            raise RuntimeError("独立 Text-CLIP 服务返回了无法解析的 JSON 响应。") from exc
        return {
            "result": [],
            "msg": f"独立 Text-CLIP 服务请求失败（HTTP {status_code}）。",
        }

    if not isinstance(payload, dict):
        if 200 <= status_code < 300:
            raise RuntimeError("独立 Text-CLIP 服务返回了非对象 JSON 响应。")
        return {
            "result": [],
            "msg": f"独立 Text-CLIP 服务请求失败（HTTP {status_code}）。",
        }

    if "result" in payload:
        if 200 <= status_code < 300:
            return payload
        message = str(
            payload.get("msg")
            or payload.get("detail")
            or f"独立 Text-CLIP 服务请求失败（HTTP {status_code}）。"
        )
        return {"result": payload.get("result", []), "msg": message}

    if "detail" in payload:
        return {
            "result": [],
            "msg": f"{payload['detail']} (HTTP {status_code})",
        }

    if 200 <= status_code < 300:
        raise RuntimeError("独立 Text-CLIP 服务返回了缺少 result 字段的响应。")
    return {"result": [], "msg": f"独立 Text-CLIP 服务请求失败（HTTP {status_code}）。"}


class TextClipProxyClient:
    def __init__(
        self,
        settings: TextClipProxySettings,
        *,
        api_key_header_name: str,
    ) -> None:
        self._settings = settings
        self._api_key_header_name = api_key_header_name
        self._endpoint_url = resolve_text_clip_endpoint_url(settings.server_url)

    def forward_request(
        self,
        raw_body: bytes,
        *,
        content_type: Optional[str],
    ) -> dict[str, Any]:
        headers = {
            "Content-Type": content_type or "application/json; charset=utf-8",
        }
        if self._settings.api_key and self._settings.api_key != "no-key":
            headers[self._api_key_header_name] = self._settings.api_key

        request = urllib.request.Request(
            self._endpoint_url,
            data=raw_body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self._settings.request_timeout_seconds,
            ) as response:
                status_code = int(getattr(response, "status", 200) or 200)
                return decode_text_clip_response(response.read(), status_code=status_code)
        except urllib.error.HTTPError as exc:
            return decode_text_clip_response(exc.read(), status_code=int(exc.code))
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, (TimeoutError, socket.timeout)):
                raise RuntimeError(
                    f"独立 Text-CLIP 服务请求超时（>{self._settings.request_timeout_seconds}s）。"
                ) from exc
            raise RuntimeError(f"独立 Text-CLIP 服务不可达：{reason}") from exc
