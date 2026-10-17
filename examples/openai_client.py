"""Reference streaming client for the vLLM OpenAI-compatible server.

    vllm serve /path/to/unified_checkpoint --served-model-name simulst
    python examples/openai_client.py --audio talk.wav --direction en-zh

The client sends a 1600 ms first chunk followed by 800 ms chunks. Each response
is appended verbatim to the conversation: an empty response represents WAIT,
and non-empty text represents EMIT.
"""

from __future__ import annotations

import argparse
import base64
import struct
import time
import wave

import numpy as np
from openai import OpenAI

SAMPLE_RATE = 16_000
INSTRUCTIONS = {
    "en-zh": "请完成下面的同声传译任务，将陆续收到的语音翻译成中文。请结合上下文边听边译，输出能够确定的译文，不必等待整句结束。",
    "zh-en": "请完成下面的同声传译任务，将陆续收到的语音翻译成英文。请结合上下文边听边译，输出能够确定的译文，不必等待整句结束。",
}


def load_wav(path: str) -> np.ndarray:
    """Read a 16 kHz PCM16 WAV as mono float32 samples."""
    with wave.open(path) as handle:
        if handle.getframerate() != SAMPLE_RATE or handle.getsampwidth() != 2:
            raise ValueError("expected a 16 kHz 16-bit PCM WAV")
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        if handle.getnchannels() > 1:
            pcm = pcm.reshape(-1, handle.getnchannels()).mean(axis=1)
    return pcm.astype(np.float32) / 32768.0


def next_chunk(waveform: np.ndarray, start: int, size: int) -> np.ndarray:
    """Take ``size`` samples and zero-pad the final chunk."""
    piece = waveform[start : start + size]
    if len(piece) < size:
        piece = np.pad(piece, (0, size - len(piece)))
    return piece.astype(np.float32)


def float_wav_base64(samples: np.ndarray) -> str:
    """Encode mono IEEE-float WAV data for an OpenAI ``input_audio`` part."""
    data = samples.astype("<f4").tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    header += b"fmt " + struct.pack(
        "<IHHIIHH", 16, 3, 1, SAMPLE_RATE, SAMPLE_RATE * 4, 4, 32
    )
    header += b"data" + struct.pack("<I", len(data))
    return base64.b64encode(header + data).decode("ascii")


class Conversation:
    def __init__(self, direction: str) -> None:
        self.direction = direction
        self.messages: list[dict] = []
        self.audio_samples = 0

    def add_chunk(self, samples: np.ndarray) -> None:
        parts = (
            [{"type": "text", "text": INSTRUCTIONS[self.direction]}]
            if not self.messages
            else []
        )
        parts.append(
            {
                "type": "input_audio",
                "input_audio": {
                    "data": float_wav_base64(samples),
                    "format": "wav",
                },
            }
        )
        self.messages.append({"role": "user", "content": parts})
        self.audio_samples += len(samples)

    def add_reply(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="simulst")
    parser.add_argument("--audio", required=True, help="16 kHz PCM16 WAV")
    parser.add_argument("--direction", choices=sorted(INSTRUCTIONS), required=True)
    parser.add_argument("--first-chunk-ms", type=int, default=1600)
    parser.add_argument("--chunk-ms", type=int, default=800)
    parser.add_argument("--window-seconds", type=float, default=25.0)
    parser.add_argument("--max-window-seconds", type=float, default=30.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    waveform = load_wav(args.audio)
    first = args.first_chunk_ms * SAMPLE_RATE // 1000
    regular = args.chunk_ms * SAMPLE_RATE // 1000
    soft = int(args.window_seconds * SAMPLE_RATE)
    hard = int(args.max_window_seconds * SAMPLE_RATE)

    conversation = Conversation(args.direction)
    outputs: list[str] = []
    cursor = 0
    while cursor < len(waveform):
        size = first if not conversation.messages else regular
        conversation.add_chunk(next_chunk(waveform, cursor, size))
        cursor += size

        started = time.monotonic()
        response = client.chat.completions.create(
            model=args.model,
            messages=conversation.messages,
            temperature=0.0,
            max_tokens=args.max_tokens,
        )
        text = response.choices[0].message.content or ""
        conversation.add_reply(text)
        elapsed_ms = (time.monotonic() - started) * 1000
        action = "EMIT" if text else "WAIT"
        seconds = min(cursor, len(waveform)) / SAMPLE_RATE
        print(f"{seconds:7.1f}s {action} {elapsed_ms:6.0f}ms  {text}", flush=True)
        if text:
            outputs.append(text.strip())

        if conversation.audio_samples >= hard or (
            conversation.audio_samples >= soft and text
        ):
            conversation = Conversation(args.direction)

    joiner = "" if args.direction.endswith("zh") else " "
    print("\n" + joiner.join(outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
