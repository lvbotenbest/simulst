import importlib
import sys
import unittest
from importlib.metadata import distribution


class PackageSmokeTest(unittest.TestCase):
    def test_top_level_import_is_lightweight(self):
        self.assertNotIn("vllm", sys.modules)
        self.assertNotIn("torch", sys.modules)

        module = importlib.import_module("simulst_mm")

        self.assertEqual(module.__name__, "simulst_mm")
        self.assertNotIn("vllm", sys.modules)
        self.assertNotIn("torch", sys.modules)

    def test_distribution_metadata(self):
        dist = distribution("simulst-mm")

        self.assertEqual(dist.metadata["Name"], "simulst-mm")
        self.assertEqual(dist.version, "0.1.0")
        self.assertEqual(dist.metadata["Requires-Python"], ">=3.11,<3.14")
        self.assertIn(
            "pokoto <40024614+lvbotenbest@users.noreply.github.com>",
            dist.metadata.get("Author-email", ""),
        )

        requirements = " ".join(dist.requires or []).replace(" ", "").lower()
        self.assertIn("vllm[audio]", requirements)
        self.assertIn(">=0.13", requirements)
        self.assertIn("<0.14", requirements)
        self.assertIn("safetensors>=0.4", requirements)
        self.assertIn("openai>=1.99.1", requirements)
        self.assertIn("numpy>=1.26", requirements)
        self.assertIn('extra=="client"', requirements)

        urls = set(dist.metadata.get_all("Project-URL") or [])
        self.assertIn(
            "Homepage, https://github.com/lvbotenbest/simulst_mm", urls
        )
        self.assertIn(
            "Source, https://github.com/lvbotenbest/simulst_mm", urls
        )

    def test_entry_points(self):
        dist = distribution("simulst-mm")
        entry_points = {
            (entry.group, entry.name): entry.value
            for entry in dist.entry_points
        }

        self.assertEqual(
            entry_points[("vllm.general_plugins", "simulst_mm")],
            "simulst_mm.plugin:register",
        )
        self.assertEqual(
            entry_points[("vllm.logits_processors", "simulst_boundary")],
            "simulst_mm.model.boundary:BoundaryGreedyLogitsProcessor",
        )

    def test_namespace_model_files_are_in_wheel(self):
        dist = distribution("simulst-mm")
        files = {str(path).replace("\\", "/") for path in (dist.files or [])}
        required = {
            "simulst_mm/chat_template.py",
            "simulst_mm/model/audio_encoder.py",
            "simulst_mm/model/audio_tower.py",
            "simulst_mm/model/boundary.py",
            "simulst_mm/model/hy_v3_speech.py",
            "simulst_mm/model/processing.py",
        }

        self.assertTrue(required <= files, required - files)


if __name__ == "__main__":
    unittest.main()
