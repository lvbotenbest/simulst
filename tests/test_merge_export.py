import unittest

from simulst_mm.export.merge_export import parse_args


class MergeExportArgumentsTest(unittest.TestCase):
    def parse(self, *extra):
        return parse_args(
            [
                "--llm",
                "/llm",
                "--audio",
                "/audio",
                "--target",
                "/target",
                *extra,
            ]
        )

    def test_openai_assets_enabled_by_default(self):
        self.assertFalse(self.parse().no_openai_assets)

    def test_openai_assets_can_be_disabled(self):
        self.assertTrue(self.parse("--no-openai-assets").no_openai_assets)


if __name__ == "__main__":
    unittest.main()
