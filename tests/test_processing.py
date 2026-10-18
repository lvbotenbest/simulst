import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("vllm"), "vLLM is not installed")
class ProcessingContractTest(unittest.TestCase):
    def test_placeholder_shape_detection(self):
        from simulst_mm.model.processing import _audio_placeholder_shape

        self.assertEqual(_audio_placeholder_shape(2, 2, 16), "compact")
        self.assertEqual(_audio_placeholder_shape(16, 2, 16), "expanded")
        self.assertEqual(_audio_placeholder_shape(0, 0, 0), "expanded")
        with self.assertRaisesRegex(ValueError, "expected either one per chunk"):
            _audio_placeholder_shape(3, 2, 16)


if __name__ == "__main__":
    unittest.main()
