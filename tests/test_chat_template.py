import json
import os
import tempfile
import unittest
from pathlib import Path

from jinja2 import Environment

from simulst_mm.chat_template import (
    ASSISTANT,
    AUDIO_CHUNK_PLACEHOLDER,
    BOS,
    EMPTY_THINK,
    GENERATION_CONFIG_FILE,
    MARKER,
    REASONING_PREFIX,
    TEMPLATE_FILE,
    TURN_END,
    USER,
    generation_config,
    unwrap_chat_template,
    wrap_chat_template,
    write_openai_assets,
)


class ChatTemplateTest(unittest.TestCase):
    def render(self, messages, *, add_generation_prompt=True):
        original = "ORIGINAL"
        template = Environment().from_string(wrap_chat_template(original))
        return template.render(
            messages=messages, add_generation_prompt=add_generation_prompt
        )

    def test_wrap_is_reversible_and_idempotent(self):
        original = "base {{ messages }}"
        wrapped = wrap_chat_template(original)
        self.assertTrue(wrapped.startswith(MARKER))
        self.assertEqual(wrap_chat_template(wrapped), wrapped)
        self.assertEqual(unwrap_chat_template(wrapped), original)
        self.assertEqual(unwrap_chat_template(original), original)

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

    def test_list_assistant_history_is_preserved(self):
        messages = [
            {"role": "user", "content": [{"type": "audio"}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "hello"}],
            },
        ]
        self.assertIn(f"{ASSISTANT}{EMPTY_THINK}hello{TURN_END}", self.render(messages))

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
    }

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

    def test_missing_special_token_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            self.make_model(model)
            config = json.loads((model / "tokenizer_config.json").read_text())
            config["added_tokens_decoder"].pop("7")
            (model / "tokenizer_config.json").write_text(json.dumps(config))

            with self.assertRaisesRegex(ValueError, "missing special tokens"):
                write_openai_assets(model)


if __name__ == "__main__":
    unittest.main()
