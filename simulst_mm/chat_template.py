"""Install OpenAI chat assets into a merged SimulST checkpoint.

The template adds an audio-aware branch for ``/v1/chat/completions`` while
preserving the checkpoint's original template for plain-string callers. Each
OpenAI audio part becomes one placeholder token; the multimodal processor later
expands it to the number of speech tokens required by that chunk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

BOS = "<｜hy_begin▁of▁sentence｜>"
REASONING_PREFIX = "<｜reasoning_mode｜>reasoning_effort:no_think"
USER = "<｜hy_User｜>"
ASSISTANT = "<｜hy_Assistant｜>"
EMPTY_THINK = "<think></think>"
TURN_END = "<｜hy_EOT｜>"
AUDIO_BEGIN = "<｜hy_place▁holder▁no▁6｜>"
AUDIO_TOKEN = "<｜hy_place▁holder▁no▁5｜>"
AUDIO_END = "<｜hy_place▁holder▁no▁8｜>"
AUDIO_CHUNK_PLACEHOLDER = f"{AUDIO_BEGIN}{AUDIO_TOKEN}{AUDIO_END}"

TEMPLATE_FILE = "chat_template.jinja"
GENERATION_CONFIG_FILE = "generation_config.json"
MARKER = "{#- simulst-mm openai chat branch v1 #}"
DEFAULT_MAX_NEW_TOKENS = 64

SIMULST_BRANCH = (
    MARKER + "\n"
    "{%- if messages and messages[0]['content'] is not string -%}\n"
    f"{{{{- '{BOS}{REASONING_PREFIX}' -}}}}\n"
    "{%- for message in messages -%}\n"
    "{%- if message['role'] == 'user' -%}\n"
    f"{{{{- '{USER}' -}}}}\n"
    "{%- for part in message['content'] -%}\n"
    "{%- if part['type'] == 'text' -%}{{- part['text'] -}}\n"
    f"{{%- elif part['type'] in ['audio', 'input_audio'] -%}}{{{{- '{AUDIO_CHUNK_PLACEHOLDER}' -}}}}\n"
    "{%- endif -%}\n"
    "{%- endfor -%}\n"
    "{%- elif message['role'] == 'assistant' -%}\n"
    f"{{{{- '{ASSISTANT}{EMPTY_THINK}' -}}}}\n"
    "{%- if message['content'] is string -%}\n"
    "{{- message['content'] -}}\n"
    "{%- else -%}\n"
    "{%- for part in message['content'] -%}\n"
    "{%- if part['type'] == 'text' -%}{{- part['text'] -}}{%- endif -%}\n"
    "{%- endfor -%}\n"
    "{%- endif -%}\n"
    f"{{{{- '{TURN_END}' -}}}}\n"
    "{%- endif -%}\n"
    "{%- endfor -%}\n"
    f"{{%- if add_generation_prompt -%}}{{{{- '{ASSISTANT}{EMPTY_THINK}' -}}}}{{%- endif -%}}\n"
    "{%- else -%}\n"
)
BRANCH_END = "\n{%- endif -%}\n"


def wrap_chat_template(original: str) -> str:
    """Prefix ``original`` with the SimulST branch; idempotent."""
    if original.startswith(MARKER):
        return original
    return SIMULST_BRANCH + original + BRANCH_END


def unwrap_chat_template(template: str) -> str:
    """Recover the original template from a wrapped one."""
    if not template.startswith(MARKER):
        return template
    return template[len(SIMULST_BRANCH) : -len(BRANCH_END)]


def generation_config(
    eos_token_id: int, turn_end_token_id: int, bos_token_id: int
) -> dict:
    """Return greedy defaults that stop on EOS or the SimulST turn marker."""
    return {
        "bos_token_id": bos_token_id,
        "eos_token_id": [eos_token_id, turn_end_token_id],
        "temperature": 0.0,
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
    }


def _replace(path: Path, text: str) -> None:
    """Replace through a fresh inode so a hard-linked source stays unchanged."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_openai_assets(model_dir: Path) -> list[str]:
    """Install the wrapped template and generation config into ``model_dir``."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    turn_end = config.get("turn_end_token_id")
    if turn_end is None:
        raise ValueError(
            f"{model_dir}/config.json has no turn_end_token_id; run merge_export"
        )

    tokenizer_config = json.loads(
        (model_dir / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    added = tokenizer_config.get("added_tokens_decoder", {})
    ids = {entry["content"]: int(token_id) for token_id, entry in added.items()}
    required = (BOS, USER, ASSISTANT, TURN_END, AUDIO_BEGIN, AUDIO_TOKEN, AUDIO_END)
    missing = [token for token in required if token not in ids]
    if missing:
        raise ValueError(f"tokenizer is missing special tokens {missing}")
    if ids[TURN_END] != int(turn_end):
        raise ValueError("turn_end_token_id in config.json does not match the tokenizer")
    if ids[AUDIO_TOKEN] != int(config.get("audio_token_id", -1)):
        raise ValueError("audio_token_id in config.json does not match the tokenizer")

    written = []
    template_path = model_dir / TEMPLATE_FILE
    original = template_path.read_text(encoding="utf-8")
    wrapped = wrap_chat_template(original)
    if wrapped != original:
        _replace(template_path, wrapped)
        written.append(TEMPLATE_FILE)

    eos = config.get("eos_token_id")
    eos = eos[0] if isinstance(eos, list) else eos
    if eos is None:
        raise ValueError(f"{model_dir}/config.json has no eos_token_id")
    payload = generation_config(int(eos), int(turn_end), ids[BOS])
    _replace(
        model_dir / GENERATION_CONFIG_FILE,
        json.dumps(payload, indent=2) + "\n",
    )
    written.append(GENERATION_CONFIG_FILE)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", required=True, help="merged checkpoint directory (modified in place)"
    )
    args = parser.parse_args(argv)
    for name in write_openai_assets(Path(args.model)):
        print(f"-- wrote {Path(args.model) / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
