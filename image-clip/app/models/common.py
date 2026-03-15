import asyncio
import os
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np

if os.name == "nt":
    import msvcrt  # type: ignore[attr-defined]

    fcntl = None  # type: ignore[assignment]
else:
    import fcntl  # type: ignore[attr-defined]

    msvcrt = None  # type: ignore[assignment]


def _as_contiguous_bgr_uint8(image: Any, context: str) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{context} expects BGR image with shape (H, W, 3), got {image.shape}")
    if image.dtype != np.uint8:
        image = image.astype(np.uint8, copy=False)
    if image.flags.c_contiguous:
        return image
    return np.ascontiguousarray(image)


def _is_lock_contention_error(exc: OSError) -> bool:
    if isinstance(exc, (BlockingIOError, PermissionError)):
        return True
    if getattr(exc, "errno", None) in {11, 13}:
        return True
    if getattr(exc, "winerror", None) in {33, 36}:
        return True
    return False


class _InterProcessFileLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Optional[Any] = None

    def acquire(self, timeout: Optional[float], blocking: bool = True) -> bool:
        if self._handle is not None:
            return True

        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if handle.tell() == 0 and self.path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()

        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        while True:
            try:
                self._try_lock(handle)
                self._handle = handle
                return True
            except OSError as exc:
                if not _is_lock_contention_error(exc):
                    handle.close()
                    raise
                if not blocking:
                    handle.close()
                    return False
                if deadline is not None and time.monotonic() >= deadline:
                    handle.close()
                    return False
                time.sleep(0.05)

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return

        try:
            if os.name == "nt":
                if msvcrt is None:
                    raise RuntimeError("msvcrt is required for Windows file locking.")
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                if fcntl is None:
                    raise RuntimeError("fcntl is required for POSIX file locking.")
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None

    @staticmethod
    def _try_lock(handle: Any) -> None:
        if os.name == "nt":
            if msvcrt is None:
                raise RuntimeError("msvcrt is required for Windows file locking.")
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return

        if fcntl is None:
            raise RuntimeError("fcntl is required for POSIX file locking.")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


class _AdmissionController:
    def __init__(self, capacity: int) -> None:
        self._capacity = max(1, int(capacity))
        self._active = 0
        self._condition = threading.Condition()

    @property
    def capacity(self) -> int:
        with self._condition:
            return self._capacity

    def acquire(self, timeout: Optional[float]) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while self._active >= self._capacity:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        return False
                    self._condition.wait(timeout=remaining)
                else:
                    self._condition.wait()
            self._active += 1
            return True

    async def acquire_async(self, timeout: Optional[float]) -> bool:
        return await asyncio.to_thread(self.acquire, timeout)

    def release(self) -> None:
        with self._condition:
            if self._active <= 0:
                return
            self._active -= 1
            self._condition.notify_all()


@dataclass(slots=True)
class _ManagedLease:
    label: str
    _callbacks: List[Callable[[], None]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _released: bool = False

    def push(self, callback: Callable[[], None]) -> None:
        with self._lock:
            if self._released:
                callback()
                return
            self._callbacks.append(callback)

    def bind_future(self, future: Future) -> None:
        future.add_done_callback(lambda _future: self.release())

    def release(self) -> None:
        callbacks: List[Callable[[], None]]
        with self._lock:
            if self._released:
                return
            self._released = True
            callbacks = list(reversed(self._callbacks))
            self._callbacks.clear()
        for callback in callbacks:
            callback()

    async def release_async(self) -> None:
        await asyncio.to_thread(self.release)


@dataclass(slots=True)
class _ClipImageTask:
    payload: Any
    future: Future
    created_at: float
    started_at: Optional[float] = None
    started_event: threading.Event = field(default_factory=threading.Event)
    cancel_requested: threading.Event = field(default_factory=threading.Event)
