import importlib.util
import sys
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STARTER_PATH = PROJECT_ROOT / "image-clip" / "starter.py"
IMAGE_CLIP_APP_DIR = str((PROJECT_ROOT / "image-clip" / "app").resolve())


class ImageClipStarterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_sys_path = list(sys.path)
        self._original_server_module = sys.modules.get("server")

    def tearDown(self) -> None:
        sys.path[:] = self._original_sys_path
        if self._original_server_module is None:
            sys.modules.pop("server", None)
        else:
            sys.modules["server"] = self._original_server_module

    def _load_starter_module(self):
        spec = importlib.util.spec_from_file_location("image_clip_starter_under_test", STARTER_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load starter module from {STARTER_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_main_inserts_app_dir_and_delegates_to_server(self) -> None:
        sys.path[:] = [entry for entry in sys.path if entry != IMAGE_CLIP_APP_DIR]
        observed_app_dirs: list[str] = []

        fake_server = types.ModuleType("server")

        def fake_run_server() -> None:
            observed_app_dirs.append(sys.path[0])

        fake_server.run_server = fake_run_server
        sys.modules["server"] = fake_server

        starter = self._load_starter_module()
        result = starter.main()

        self.assertEqual(0, result)
        self.assertEqual([IMAGE_CLIP_APP_DIR], observed_app_dirs)
        self.assertEqual(IMAGE_CLIP_APP_DIR, sys.path[0])

    def test_main_does_not_duplicate_app_dir(self) -> None:
        sys.path[:] = [IMAGE_CLIP_APP_DIR, *[entry for entry in sys.path if entry != IMAGE_CLIP_APP_DIR]]
        call_count = 0

        fake_server = types.ModuleType("server")

        def fake_run_server() -> None:
            nonlocal call_count
            call_count += 1

        fake_server.run_server = fake_run_server
        sys.modules["server"] = fake_server

        starter = self._load_starter_module()
        starter.main()

        self.assertEqual(1, call_count)
        self.assertEqual(1, sum(1 for entry in sys.path if entry == IMAGE_CLIP_APP_DIR))
