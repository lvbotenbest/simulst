import copy
import importlib.util
import io
import json
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


CLIENT_PATH = Path(__file__).parents[1] / "examples" / "openai_client.py"


def load_client_module():
    numpy = types.ModuleType("numpy")
    numpy.ndarray = object
    openai = types.ModuleType("openai")
    openai.OpenAI = object
    with patch.dict(sys.modules, {"numpy": numpy, "openai": openai}):
        spec = importlib.util.spec_from_file_location("simulst_openai_client", CLIENT_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class FakeRawResponse:
    def __init__(self, content, message_content):
        self.content = content
        self.message_content = message_content

    def parse(self):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.message_content)
                )
            ]
        )


class FakeCompletions:
    def __init__(self, responses):
        self.with_raw_response = self
        self.responses = iter(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        return next(self.responses)


class OpenAIClientProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = load_client_module()

    def test_output_mode_is_forwarded_to_chat_template(self):
        self.assertEqual(
            self.client.chat_template_kwargs("bilingual"),
            {"simulst_output_mode": "bilingual"},
        )
        self.assertEqual(
            self.client.chat_template_kwargs("translation"),
            {"simulst_output_mode": "translation"},
        )

    def test_bilingual_wait_and_emit(self):
        self.assertFalse(
            self.client.reply_is_emit('{"src":"","tgt":""}', "bilingual")
        )
        self.assertTrue(
            self.client.reply_is_emit('{"src":"hello","tgt":""}', "bilingual")
        )
        self.assertTrue(
            self.client.reply_is_emit('{"src":"","tgt":"你好"}', "bilingual")
        )

    def test_translation_wait_and_emit(self):
        self.assertFalse(self.client.reply_is_emit("", "translation"))
        self.assertTrue(self.client.reply_is_emit("translated", "translation"))

    def test_bilingual_reply_must_follow_src_tgt_schema(self):
        invalid = (
            "not json",
            "[]",
            '{"src":"only"}',
            '{"src":1,"tgt":"translated"}',
        )
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.client.reply_is_emit(text, "bilingual")

    def test_raw_json_bytes_are_not_reserialized(self):
        stream = io.BytesIO()
        payload = b'{ "choices" : [ { "message": {"content":"x"} } ] }'
        self.client.write_raw_json(payload, stream)
        self.assertEqual(stream.getvalue(), payload + b"\n")

    def test_bilingual_control_token_is_verified(self):
        response = FakeHTTPResponse(
            {"tokens": [self.client.BILINGUAL_CONTROL_TOKEN_ID]}
        )
        with patch.object(self.client.urllib.request, "urlopen", return_value=response):
            self.assertTrue(
                self.client.verify_bilingual_control_token(
                    "http://127.0.0.1:8000/v1", "simulst"
                )
            )

    def test_bilingual_control_token_mismatch_fails(self):
        response = FakeHTTPResponse({"tokens": [1, 2]})
        with patch.object(self.client.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, r"expected \[120808\]"):
                self.client.verify_bilingual_control_token(
                    "http://127.0.0.1:8000/v1", "simulst"
                )

    def test_unavailable_tokenizer_endpoint_is_nonfatal(self):
        with patch.object(
            self.client.urllib.request,
            "urlopen",
            side_effect=self.client.urllib.error.URLError("unavailable"),
        ), patch.object(self.client.sys, "stderr", io.StringIO()) as stderr:
            self.assertFalse(
                self.client.verify_bilingual_control_token(
                    "http://127.0.0.1:8000/v1", "simulst"
                )
            )
            self.assertIn("could not verify", stderr.getvalue())

    def test_main_preserves_raw_json_and_assistant_history(self):
        wait = '{"src":"","tgt":""}'
        emit = '{"src":"hello","tgt":"你好"}'
        completions = FakeCompletions(
            [
                FakeRawResponse(b'{"turn":1}', wait),
                FakeRawResponse(b'{"turn":2}', emit),
            ]
        )
        fake_openai = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        raw_bodies = []

        class Sized:
            def __init__(self, size):
                self.size = size

            def __len__(self):
                return self.size

        with patch.object(
            self.client.sys,
            "argv",
            ["openai_client.py", "--audio", "talk.wav", "--direction", "en-zh"],
        ), patch.object(self.client, "OpenAI", return_value=fake_openai), patch.object(
            self.client, "verify_bilingual_control_token", return_value=True
        ), patch.object(
            self.client, "load_wav", return_value=Sized(38_400)
        ), patch.object(
            self.client, "next_chunk", side_effect=lambda _waveform, _start, size: Sized(size)
        ), patch.object(
            self.client, "float_wav_base64", return_value="audio"
        ), patch.object(
            self.client, "write_raw_json", side_effect=raw_bodies.append
        ), patch.object(
            self.client.sys, "stderr", io.StringIO()
        ):
            self.assertEqual(self.client.main(), 0)

        self.assertEqual(raw_bodies, [b'{"turn":1}', b'{"turn":2}'])
        self.assertEqual(len(completions.requests), 2)
        second_messages = completions.requests[1]["messages"]
        self.assertEqual(second_messages[1], {"role": "assistant", "content": wait})
        self.assertEqual(
            completions.requests[0]["extra_body"],
            {
                "chat_template_kwargs": {
                    "simulst_output_mode": "bilingual"
                }
            },
        )
        self.assertEqual(completions.requests[0]["max_tokens"], 128)


if __name__ == "__main__":
    unittest.main()
