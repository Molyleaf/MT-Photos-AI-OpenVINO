from typing import TYPE_CHECKING

from .constants import CLIP_EMBEDDING_DIMS, CLIP_IMAGE_RESOLUTION, CONTEXT_LENGTH, MODEL_NAME

if TYPE_CHECKING:
    from .runtime import AIModels

__all__ = [
    "AIModels",
    "MODEL_NAME",
    "CLIP_EMBEDDING_DIMS",
    "CONTEXT_LENGTH",
    "CLIP_IMAGE_RESOLUTION",
]


def __getattr__(name: str):
    if name == "AIModels":
        from .runtime import AIModels

        return AIModels
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
