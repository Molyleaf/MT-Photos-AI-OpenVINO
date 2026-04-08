import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import server


class _FakeManager:
    def __init__(self) -> None:
        self.release_calls = 0

    def release_models_for_restart(self) -> None:
        self.release_calls += 1


class _FakeRequest:
    def __init__(self, path: str) -> None:
        self.url = SimpleNamespace(path=path)


class ServerRestartRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._previous_models_instance = server.models_instance

    def tearDown(self) -> None:
        server.models_instance = self._previous_models_instance

    def test_restart_paths_are_registered(self) -> None:
        route_paths = {
            route.path
            for route in server.app.routes
            if "POST" in getattr(route, "methods", set())
        }

        self.assertIn("/restart", route_paths)
        self.assertIn("/restart_v2", route_paths)
        self.assertIn("/restartV2", route_paths)
        self.assertIn("/restartv2", route_paths)

    def test_restart_paths_share_the_same_release_logic(self) -> None:
        manager = _FakeManager()
        server.models_instance = manager

        for path in ("/restart", "/restart_v2", "/restartV2", "/restartv2"):
            response = asyncio.run(server.restart_non_text_models(_FakeRequest(path)))
            self.assertEqual({"result": "pass"}, response)

        self.assertEqual(4, manager.release_calls)

    def test_restart_path_succeeds_without_manager_instance(self) -> None:
        server.models_instance = None

        response = asyncio.run(server.restart_non_text_models(_FakeRequest("/restart")))

        self.assertEqual({"result": "pass"}, response)


if __name__ == "__main__":
    unittest.main()
