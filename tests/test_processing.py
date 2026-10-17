import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("vllm"), "vLLM is not installed")
class ProcessingContractTest(unittest.TestCase):
    def test_placeholder_shape_detection(self):
        from simulst_mm.model.processing import _is_audio_update_applied

        self.assertFalse(_is_audio_update_applied(2, 2, 16))
        self.assertTrue(_is_audio_update_applied(16, 2, 16))
        self.assertTrue(_is_audio_update_applied(0, 0, 0))
        with self.assertRaisesRegex(ValueError, "expected either one per chunk"):
            _is_audio_update_applied(3, 2, 16)


if __name__ == "__main__":
    unittest.main()
