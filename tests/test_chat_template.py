import json
import os
import tempfile
import unittest
from pathlib import Path

from jinja2 import Environment

from simulst_mm.chat_template import (
    ASSISTANT,
    AUDIO_CHUNK_PLACEHOLDER,
    BILINGUAL_CONTROL_TOKEN,
    BILINGUAL_CONTROL_TOKEN_ID,
    BOS,
    EMPTY_THINK,
    GENERATION_CONFIG_FILE,
    LEGACY_MARKER,
    LEGACY_SIMULST_BRANCH,
    MARKER,
    REASONING_PREFIX,
    TEMPLATE_FILE,
    TURN_END,
    USER,
    assets_installed,
    generation_config,
    missing_assets_error,
    unwrap_chat_template,
    wrap_chat_template,
    write_openai_assets,
)


class ChatTemplateTest(unittest.TestCase):
    def render(self, messages, *, add_generation_prompt=True, **kwargs):
        original = "ORIGINAL"
        template = Environment().from_string(wrap_chat_template(original))
        return template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    def test_wrap_is_reversible_and_idempotent(self):
        original = "base {{ messages }}"
        wrapped = wrap_chat_template(original)
        self.assertTrue(wrapped.startswith(MARKER))
        self.assertEqual(wrap_chat_template(wrapped), wrapped)
        self.assertEqual(unwrap_chat_template(wrapped), original)
        self.assertEqual(unwrap_chat_template(original), original)

    def test_legacy_template_is_migrated(self):
        original = "base {{ messages }}"
        legacy = LEGACY_SIMULST_BRANCH + original + "\n{%- endif -%}\n"
        wrapped = wrap_chat_template(legacy)
        self.assertTrue(wrapped.startswith(MARKER))
        self.assertEqual(unwrap_chat_template(wrapped), original)

    def test_string_assistant_history_is_preserved(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "translate"},
                    {"type": "audio"},
                ],
            },
            {"role": "assistant", "content": "translated text"},
            {"role": "user", "content": [{"type": "input_audio"}]},
            {"role": "assistant", "content": ""},
        ]
        rendered = self.render(messages)
        expected = (
            BOS
            + REASONING_PREFIX
            + USER
            + "translate"
            + AUDIO_CHUNK_PLACEHOLDER
            + ASSISTANT
            + EMPTY_THINK
            + "translated text"
            + TURN_END
            + USER
            + AUDIO_CHUNK_PLACEHOLDER
            + ASSISTANT
            + EMPTY_THINK
            + TURN_END
            + ASSISTANT
            + EMPTY_THINK
        )
        self.assertEqual(rendered, expected)

    def test_malformed_legacy_template_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "wrapper is malformed"):
            wrap_chat_template(LEGACY_MARKER + "damaged")

    def test_list_assistant_history_is_preserved(self):
        messages = [
            {"role": "user", "content": [{"type": "audio"}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "hello"}],
            },
        ]
        self.assertIn(f"{ASSISTANT}{EMPTY_THINK}hello{TURN_END}", self.render(messages))

    def test_bilingual_control_token_is_last_generation_prompt_token(self):
        messages = [
            {"role": "user", "content": [{"type": "audio"}]},
            {"role": "assistant", "content": "history"},
            {"role": "user", "content": [{"type": "audio"}]},
        ]
        rendered = self.render(messages, simulst_output_mode="bilingual")
        self.assertTrue(
            rendered.endswith(f"{ASSISTANT}{EMPTY_THINK}{BILINGUAL_CONTROL_TOKEN}")
        )
        self.assertEqual(rendered.count(BILINGUAL_CONTROL_TOKEN), 1)

    def test_translation_mode_does_not_add_control_token(self):
        messages = [{"role": "user", "content": [{"type": "audio"}]}]
        self.assertNotIn(
            BILINGUAL_CONTROL_TOKEN,
            self.render(messages, simulst_output_mode="translation"),
        )

    def test_plain_string_conversation_uses_original_template(self):
        original = "{{ messages[0]['content'] }}"
        template = Environment().from_string(wrap_chat_template(original))
        self.assertEqual(
            template.render(
                messages=[{"role": "user", "content": "plain"}],
                add_generation_prompt=False,
            ),
            "plain",
        )

    def test_generation_config(self):
        self.assertEqual(
            generation_config(2, 3, 1),
            {
                "bos_token_id": 1,
                "eos_token_id": [2, 3],
                "temperature": 0.0,
                "max_new_tokens": 64,
            },
        )


class OpenAIAssetsTest(unittest.TestCase):
    TOKENS = {
        BOS: 1,
        USER: 2,
        ASSISTANT: 3,
        TURN_END: 4,
        "<｜hy_place▁holder▁no▁6｜>": 5,
        "<｜hy_place▁holder▁no▁5｜>": 6,
        "<｜hy_place▁holder▁no▁8｜>": 7,
        BILINGUAL_CONTROL_TOKEN: BILINGUAL_CONTROL_TOKEN_ID,
    }

    def setUp(self):
        assets_installed.cache_clear()

    def make_model(self, root: Path):
        (root / "config.json").write_text(
            json.dumps(
                {
                    "eos_token_id": 8,
                    "turn_end_token_id": 4,
                    "audio_token_id": 6,
                }
            ),
            encoding="utf-8",
        )
        (root / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "added_tokens_decoder": {
                        str(token_id): {"content": token}
                        for token, token_id in self.TOKENS.items()
                    }
                }
            ),
            encoding="utf-8",
        )
        (root / TEMPLATE_FILE).write_text("BASE", encoding="utf-8")

    def test_writes_assets_without_mutating_hard_link_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source_template.jinja"
            source.write_text("BASE", encoding="utf-8")
            model = root / "model"
            model.mkdir()
            self.make_model(model)
            (model / TEMPLATE_FILE).unlink()
            os.link(source, model / TEMPLATE_FILE)

            written = write_openai_assets(model)

            self.assertEqual(written, [TEMPLATE_FILE, GENERATION_CONFIG_FILE])
            self.assertEqual(source.read_text(encoding="utf-8"), "BASE")
            self.assertTrue((model / TEMPLATE_FILE).read_text().startswith(MARKER))
            payload = json.loads((model / GENERATION_CONFIG_FILE).read_text())
            self.assertEqual(payload["eos_token_id"], [8, 4])
            self.assertTrue(assets_installed(str(model)))

    def test_missing_assets_reports_repair_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(assets_installed(tmp))
            message = missing_assets_error(tmp)
            self.assertIn("python -m simulst_mm.chat_template --model", message)
            self.assertIn(tmp, message)

    def test_missing_eos_token_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            self.make_model(model)
            config = json.loads((model / "config.json").read_text())
            config.pop("eos_token_id")
            (model / "config.json").write_text(json.dumps(config))

            with self.assertRaisesRegex(ValueError, "has no eos_token_id"):
                write_openai_assets(model)

    def test_missing_special_token_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            self.make_model(model)
            config = json.loads((model / "tokenizer_config.json").read_text())
            config["added_tokens_decoder"].pop("7")
            (model / "tokenizer_config.json").write_text(json.dumps(config))

            with self.assertRaisesRegex(ValueError, "missing special tokens"):
                write_openai_assets(model)

    def test_missing_bilingual_control_token_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            self.make_model(model)
            config = json.loads((model / "tokenizer_config.json").read_text())
            config["added_tokens_decoder"].pop(str(BILINGUAL_CONTROL_TOKEN_ID))
            (model / "tokenizer_config.json").write_text(json.dumps(config))

            with self.assertRaisesRegex(ValueError, "no bilingual control token"):
                write_openai_assets(model)

    def test_conflicting_bilingual_control_token_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            self.make_model(model)
            config = json.loads((model / "tokenizer_config.json").read_text())
            config["added_tokens_decoder"][str(BILINGUAL_CONTROL_TOKEN_ID)][
                "content"
            ] = "wrong"
            (model / "tokenizer_config.json").write_text(json.dumps(config))

            with self.assertRaisesRegex(ValueError, "token 120808"):
                write_openai_assets(model)


if __name__ == "__main__":
    unittest.main()
