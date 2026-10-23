import sys
import types
import unittest
from unittest.mock import patch

import simulst_mm.plugin as plugin


class FakeRegistry:
    supported = {plugin.LANGUAGE_ARCHITECTURE}
    registrations = []

    @classmethod
    def get_supported_archs(cls):
        return list(cls.supported)

    @classmethod
    def register_model(cls, architecture, model_path):
        cls.registrations.append((architecture, model_path))
        cls.supported.add(architecture)


class PluginTest(unittest.TestCase):
    def setUp(self):
        plugin._registered = False
        FakeRegistry.supported = {plugin.LANGUAGE_ARCHITECTURE}
        FakeRegistry.registrations = []
        self.vllm = types.ModuleType("vllm")
        self.vllm.ModelRegistry = FakeRegistry

    def test_registers_once_after_compatibility_checks(self):
        with patch.dict(sys.modules, {"vllm": self.vllm}), patch.object(
            plugin, "version", return_value="0.31.0"
        ):
            plugin.register()
            plugin.register()

        self.assertEqual(
            FakeRegistry.registrations,
            [(plugin.ARCHITECTURE, plugin.MODEL_PATH)],
        )

    def test_rejects_wrong_vllm_version_without_registering(self):
        with patch.dict(sys.modules, {"vllm": self.vllm}), patch.object(
            plugin, "version", return_value="0.30.0"
        ):
            with self.assertRaisesRegex(RuntimeError, "requires upstream vLLM >=0.31"):
                plugin.register()

        self.assertEqual(FakeRegistry.registrations, [])
        self.assertFalse(plugin._registered)

    def test_accepts_newer_vllm_when_required_architecture_exists(self):
        for installed_version in ("0.32.0", "1.0.0"):
            with self.subTest(installed_version=installed_version):
                plugin._registered = False
                FakeRegistry.supported = {plugin.LANGUAGE_ARCHITECTURE}
                FakeRegistry.registrations = []
                with patch.dict(sys.modules, {"vllm": self.vllm}), patch.object(
                    plugin, "version", return_value=installed_version
                ):
                    plugin.register()
                self.assertEqual(
                    FakeRegistry.registrations,
                    [(plugin.ARCHITECTURE, plugin.MODEL_PATH)],
                )

    def test_rejects_vllm_without_hyv3(self):
        FakeRegistry.supported = set()
        with patch.dict(sys.modules, {"vllm": self.vllm}), patch.object(
            plugin, "version", return_value="0.31.0"
        ):
            with self.assertRaisesRegex(RuntimeError, plugin.LANGUAGE_ARCHITECTURE):
                plugin.register()

        self.assertEqual(FakeRegistry.registrations, [])
        self.assertFalse(plugin._registered)

    def test_reports_missing_vllm(self):
        with patch.dict(sys.modules, {"vllm": None}):
            with self.assertRaisesRegex(ImportError, r"simulst\[server\]"):
                plugin.register()


if __name__ == "__main__":
    unittest.main()
