from pathlib import Path
import sys


IMAGE_CLIP_ROOT = Path(__file__).resolve().parent
IMAGE_CLIP_APP_DIR = IMAGE_CLIP_ROOT / "app"


def _ensure_app_dir_on_sys_path() -> None:
    app_dir = str(IMAGE_CLIP_APP_DIR)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)


def main() -> int:
    _ensure_app_dir_on_sys_path()

    from server import run_server

    run_server()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
