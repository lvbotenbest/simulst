"""Audio tower: encoder plus the FSK adapter that lands in LLM embedding space.

Checkpoint-compatible with the deployed ``audio_frontend.safetensors``: the two
submodules are named ``speech_encoder`` and ``speech_adapter``, and the adapter
keeps its ``layers.0`` / ``layers.3`` / ``layers.4`` numbering, so the 403
trained tensors load by name with nothing renamed.

The frame arithmetic is duplicated here as pure functions rather than imported,
for the same reason the encoder is vendored -- this package must stand on its own
inside a plain ``vllm serve``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from simulst_mm.model.audio_encoder import (
    AudioEncoder,
    AudioEncoderConfig,
    encoder_frames_from_mel_frames,
)

SAMPLE_RATE = 16_000
HOP_LENGTH = 160
MAX_SAMPLES = 480_000


def mel_frames_from_num_samples(
    num_samples: int,
    hop_length: int = HOP_LENGTH,
    max_samples: int = MAX_SAMPLES,
) -> int:
    """Waveform samples to mel frames, rounding a partial frame up."""
    if num_samples < 0:
        raise ValueError(f"num_samples must be non-negative, got {num_samples}")
    samples = min(num_samples, max_samples)
    return (samples + hop_length - 1) // hop_length


def speech_tokens_from_num_samples(
    num_samples: int,
    fsk_size: int = 2,
    max_samples: int = MAX_SAMPLES,
    hop_length: int = HOP_LENGTH,
) -> int:
    """Placeholder count a waveform of this length expands to.

    This is the number the prompt builder already uses to size its run of
    ``audio_token_id``, and the number vLLM checks the embeddings against, so
    the two sides must agree exactly or the request is rejected.
    """
    if fsk_size < 1:
        raise ValueError(f"fsk_size must be at least 1, got {fsk_size}")
    mel_frames = mel_frames_from_num_samples(num_samples, hop_length, max_samples)
    return (encoder_frames_from_mel_frames(mel_frames) + fsk_size - 1) // fsk_size


class FSKAdapter(nn.Module):
    """Stack adjacent encoder frames and project them into the LLM space."""

    def __init__(self, encoder_dim: int = 2048, llm_dim: int = 2048, fsk_size: int = 2) -> None:
        super().__init__()
        if encoder_dim < 1 or llm_dim < 1:
            raise ValueError("encoder_dim and llm_dim must be positive")
        if fsk_size < 1:
            raise ValueError(f"fsk_size must be at least 1, got {fsk_size}")

        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
        self.stack_size = fsk_size
        # Layout and numbering are fixed by existing speech_adapter checkpoints.
        self.layers = nn.Sequential(
            nn.Linear(encoder_dim * fsk_size, llm_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Linear(llm_dim, llm_dim, bias=True),
            nn.LayerNorm(llm_dim, eps=1e-5, bias=True),
        )

    @property
    def fc_in(self) -> nn.Linear:
        return self.layers[0]

    def _stack_frames(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                f"hidden_states must have shape (B, T, D), got {tuple(hidden_states.shape)}"
            )
        if hidden_states.size(-1) != self.encoder_dim:
            raise ValueError(
                f"encoder feature dimension must be {self.encoder_dim}, "
                f"got {hidden_states.size(-1)}"
            )

        batch_size, seq_len, hidden_size = hidden_states.shape
        remainder = seq_len % self.stack_size
        if remainder:
            hidden_states = F.pad(hidden_states, (0, 0, 0, self.stack_size - remainder))
            seq_len = hidden_states.size(1)
        return hidden_states.reshape(
            batch_size, seq_len // self.stack_size, hidden_size * self.stack_size
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if padding_mask is not None:
            if padding_mask.shape != hidden_states.shape[:2]:
                raise ValueError(
                    f"padding_mask must have shape {tuple(hidden_states.shape[:2])}, "
                    f"got {tuple(padding_mask.shape)}"
                )
            hidden_states = hidden_states.masked_fill(
                padding_mask.to(torch.bool).unsqueeze(-1), 0.0
            )

        stacked = self._stack_frames(hidden_states)
        return self.layers(stacked.to(dtype=self.fc_in.weight.dtype))


class AudioTower(nn.Module):
    """Packed mel features to LLM-space speech embeddings."""

    def __init__(
        self,
        encoder_config: AudioEncoderConfig,
        llm_dim: int = 2048,
        fsk_size: int = 2,
    ) -> None:
        super().__init__()
        self.encoder_dim = encoder_config.output_dim
        self.fsk_size = fsk_size
        self.speech_encoder = AudioEncoder(encoder_config)
        self.speech_adapter = FSKAdapter(self.encoder_dim, llm_dim, fsk_size)

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode packed mel features.

        Returns ``(embeddings, speech_lens)``: embeddings are concatenated in
        item order with no padding, ``speech_lens`` says how to split them.
        """
        if input_features.ndim != 2:
            raise ValueError(
                f"packed input_features must be 2D, got shape {tuple(input_features.shape)}"
            )
        feature_lens = feature_lens.to(device=input_features.device, dtype=torch.long)

        encoder_dtype = next(self.speech_encoder.parameters()).dtype
        hidden_states = self.speech_encoder(
            input_features.to(dtype=encoder_dtype), feature_lens=feature_lens
        )

        encoder_lens = torch.tensor(
            [encoder_frames_from_mel_frames(int(length)) for length in feature_lens],
            dtype=torch.long,
            device=hidden_states.device,
        )
        if hidden_states.ndim != 2 or hidden_states.size(0) != int(encoder_lens.sum()):
            raise RuntimeError(
                f"audio encoder returned shape {tuple(hidden_states.shape)}, expected "
                f"({int(encoder_lens.sum())}, {self.encoder_dim})"
            )

        # Pad to a rectangle for the adapter, mask the padding so it cannot
        # leak through the frame stacking, then drop the padded tokens again.
        chunks = torch.split(hidden_states, encoder_lens.tolist(), dim=0)
        padded = nn.utils.rnn.pad_sequence(chunks, batch_first=True)
        positions = torch.arange(padded.size(1), device=padded.device)
        padding_mask = positions.unsqueeze(0) >= encoder_lens.unsqueeze(1)
        adapted = self.speech_adapter(padded, padding_mask)

        speech_lens = (encoder_lens + self.fsk_size - 1) // self.fsk_size
        token_positions = torch.arange(adapted.size(1), device=adapted.device)
        keep = token_positions.unsqueeze(0) < speech_lens.unsqueeze(1)
        return adapted[keep], speech_lens


__all__ = [
    "AudioTower",
    "FSKAdapter",
    "HOP_LENGTH",
    "MAX_SAMPLES",
    "SAMPLE_RATE",
    "mel_frames_from_num_samples",
    "speech_tokens_from_num_samples",
]
