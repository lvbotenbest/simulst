"""Packed-input audio encoder.

Vendored rather than imported. The training code reaches this architecture
through ``hytuner``, which lives in a read-only shared tree and cannot be
pip-installed; depending on it would mean ``vllm serve`` only works on machines
that happen to have that tree mounted. Everything here is a plain
``nn.Module`` -- no ``PreTrainedModel``, no ``from_pretrained`` -- because vLLM
constructs the model itself and then loads weights through ``AutoWeightsLoader``.

The contract is unusual in one way: ``forward`` takes a *packed* mel tensor
shaped ``(num_mel_bins, sum(feature_lens))`` plus the per-item lengths, instead
of a padded batch. Internally it re-chunks on a ``n_window * 2`` grid, runs the
three strided convolutions, and attends within blocks delimited by
``cu_seqlens``, so items never attend across each other.

Numerics must match the deployed frontend exactly, so the arithmetic below is a
faithful transcription: same ``_eager_attention_forward`` fallback, same fp32
softmax, same float32 sinusoid table cast once to the module dtype, same gelu.
``bench/verify_parity.py`` checks that bit-for-bit against the deployed tower.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def encoder_frames_from_mel_frames(mel_frames: int) -> int:
    """Encoder output frames for an unpadded mel sequence.

    Closed form of the three stride-2 convolutions applied on the
    ``n_window * 2 = 100`` frame grid: whole windows contribute 13 frames each,
    and the remainder is pushed through the same three halvings.
    """
    if mel_frames < 0:
        raise ValueError(f"mel_frames must be non-negative, got {mel_frames}")

    remainder = mel_frames % 100
    after_first_conv = (remainder - 1) // 2 + 1
    tail_frames = ((after_first_conv - 1) // 2 + 1 - 1) // 2 + 1
    return tail_frames + (mel_frames // 100) * 13


def feature_extract_output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
    """Tensor form of [`encoder_frames_from_mel_frames`], used inside forward."""
    leave = input_lengths % 100
    feat_lengths = (leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13


class AudioEncoderConfig:
    """Shape and behaviour of the audio encoder.

    Deliberately not a ``PretrainedConfig``: it is built from a plain dict in
    the merged checkpoint's ``config.json``, and unknown keys are ignored so a
    config carrying extra HF boilerplate still loads.
    """

    def __init__(
        self,
        num_mel_bins: int = 128,
        encoder_layers: int = 24,
        encoder_attention_heads: int = 16,
        encoder_ffn_dim: int = 4096,
        d_model: int = 1024,
        attention_dropout: float = 0.0,
        max_source_positions: int = 1500,
        n_window: int = 50,
        n_window_infer: int = 800,
        conv_chunksize: int = 500,
        downsample_hidden_size: int = 480,
        output_dim: int = 2048,
        attn_implementation: str = "sdpa",
        **ignored: object,
    ) -> None:
        self.num_mel_bins = num_mel_bins
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.d_model = d_model
        self.attention_dropout = attention_dropout
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize
        self.downsample_hidden_size = downsample_hidden_size
        self.output_dim = output_dim
        # The deployed frontend runs whatever transformers picks by default,
        # which on this install is sdpa. Pinning it keeps the served model on
        # the same kernel rather than whatever a future default becomes.
        self.attn_implementation = attn_implementation

    @classmethod
    def from_dict(cls, raw: dict) -> "AudioEncoderConfig":
        return cls(**{key: value for key, value in raw.items() if not key.startswith("_")})


def _eager_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> torch.Tensor:
    weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask[..., : key.shape[-2]]
    # Softmax in fp32 regardless of activation dtype, then back down.
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(weights, value).transpose(1, 2).contiguous()


def _sdpa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> torch.Tensor:
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        scale=scaling,
        is_causal=False,
    )
    return output.transpose(1, 2).contiguous()


_ATTENTION_IMPLS: dict[str, Callable[..., torch.Tensor]] = {
    "eager": _eager_attention,
    "sdpa": _sdpa_attention,
}


class AudioAttention(nn.Module):
    """Full (non-causal) self-attention over a single packed sequence.

    The whole packed sequence is one "row": ``hidden_states`` is 2D and the
    block structure lives entirely in ``attention_mask``, which is built once
    per forward from ``cu_seqlens``.
    """

    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        if self.embed_dim % self.num_heads:
            raise ValueError(
                f"d_model ({self.embed_dim}) must be divisible by "
                f"encoder_attention_heads ({self.num_heads})"
            )
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.attn_implementation = config.attn_implementation
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        seq_length = hidden_states.size(0)
        shape = (seq_length, self.num_heads, self.head_dim)
        query = self.q_proj(hidden_states).reshape(shape).transpose(0, 1).unsqueeze(0)
        key = self.k_proj(hidden_states).reshape(shape).transpose(0, 1).unsqueeze(0)
        value = self.v_proj(hidden_states).reshape(shape).transpose(0, 1).unsqueeze(0)

        attention = _ATTENTION_IMPLS.get(self.attn_implementation, _sdpa_attention)
        output = attention(query, key, value, attention_mask, self.scaling)
        return self.out_proj(output.reshape(seq_length, self.embed_dim).contiguous())


class AudioEncoderLayer(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        self.self_attn = AudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, attention_mask)

        residual = hidden_states
        hidden_states = self.fc2(F.gelu(self.fc1(self.final_layer_norm(hidden_states))))
        return residual + hidden_states


class SinusoidsPositionEmbedding(nn.Module):
    """Fixed sin/cos table.

    Non-persistent on purpose: it is absent from the checkpoint and recomputed
    here. Built in float32 and cast once, matching the deployed frontend, where
    ``tower.to(dtype=...)`` casts the float32 table in the same single step.
    """

    def __init__(self, length: int, channels: int, max_timescale: int = 10_000) -> None:
        super().__init__()
        if channels % 2:
            raise ValueError("SinusoidsPositionEmbedding requires an even channel count")
        increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length).float()[:, None] * inv_timescales[None, :]
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seq_len: int) -> torch.Tensor:
        return self.positional_embedding[:seq_len]


class AudioEncoder(nn.Module):
    """Mel features to encoder hidden states, on a packed variable-length input."""

    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        embed_dim = config.d_model
        self.config = config
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.n_window = config.n_window
        self.n_window_infer = config.n_window_infer
        self.conv_chunksize = config.conv_chunksize

        self.positional_embedding = SinusoidsPositionEmbedding(
            self.max_source_positions, embed_dim
        )
        self.layers = nn.ModuleList(
            AudioEncoderLayer(config) for _ in range(config.encoder_layers)
        )
        self.ln_post = nn.LayerNorm(embed_dim)
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(
            config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1
        )
        self.conv2d3 = nn.Conv2d(
            config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1
        )
        conv_frequency = (((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * conv_frequency, embed_dim, bias=False
        )
        self.proj1 = nn.Linear(embed_dim, embed_dim)
        self.proj2 = nn.Linear(embed_dim, config.output_dim)
        # Unused at inference, kept so the shape math reads like the original.
        self.embed_scale = 1.0 * math.sqrt(embed_dim) ** 0

    def _block_attention_mask(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        """Additive mask allowing attention only inside each ``cu_seqlens`` block."""
        seq_length = hidden_states.size(0)
        mask = torch.full(
            (1, 1, seq_length, seq_length),
            torch.finfo(hidden_states.dtype).min,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        for index in range(1, len(cu_seqlens)):
            start, end = int(cu_seqlens[index - 1]), int(cu_seqlens[index])
            mask[..., start:end, start:end] = 0
        return mask

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a packed mel tensor.

        Args:
            input_features: ``(num_mel_bins, sum(feature_lens))``.
            feature_lens: ``(num_items,)`` unpadded mel frame counts.

        Returns:
            ``(sum(encoder_lens), output_dim)`` hidden states in item order.
        """
        if input_features.ndim != 2 or input_features.size(0) != self.num_mel_bins:
            raise ValueError(
                f"input_features must have shape ({self.num_mel_bins}, sum(feature_lens)), "
                f"got {tuple(input_features.shape)}"
            )
        feature_lens = feature_lens.to(device=input_features.device, dtype=torch.long)
        if feature_lens.ndim != 1 or feature_lens.numel() == 0 or torch.any(feature_lens <= 0):
            raise ValueError("feature_lens must be a non-empty 1D tensor of positive lengths")
        if int(feature_lens.sum()) != input_features.size(1):
            raise ValueError(
                "sum(feature_lens) must equal packed mel length: "
                f"{int(feature_lens.sum())} != {input_features.size(1)}"
            )

        aftercnn_lens = feature_extract_output_lengths(feature_lens)

        # Re-chunk onto the n_window * 2 grid. A tail chunk keeps its true
        # length so the convolutions see exactly the frames that exist.
        window = self.n_window * 2
        chunk_num = torch.ceil(feature_lens / window).long()
        chunk_lengths = torch.full(
            (int(chunk_num.sum()),), window, dtype=torch.long, device=feature_lens.device
        )
        tail_indices = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_indices] = feature_lens % window
        chunk_lengths[chunk_lengths == 0] = window

        chunks = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded = nn.utils.rnn.pad_sequence(chunks, batch_first=True).transpose(1, 2)
        chunk_output_lens = feature_extract_output_lengths(chunk_lengths)
        valid_mask = nn.utils.rnn.pad_sequence(
            [
                torch.ones(int(length), dtype=torch.bool, device=padded.device)
                for length in chunk_output_lens
            ],
            batch_first=True,
        )

        padded = padded.unsqueeze(1)
        encoded_chunks = []
        for chunk in padded.split(self.conv_chunksize, dim=0):
            chunk = F.gelu(self.conv2d1(chunk))
            chunk = F.gelu(self.conv2d2(chunk))
            encoded_chunks.append(F.gelu(self.conv2d3(chunk)))
        padded_embed = torch.cat(encoded_chunks, dim=0)

        batch, channels, frequency, time = padded_embed.shape
        padded_embed = self.conv_out(
            padded_embed.permute(0, 3, 1, 2).contiguous().view(batch, time, channels * frequency)
        )
        positions = (
            self.positional_embedding(padded_embed.size(1))
            .unsqueeze(0)
            .to(device=padded_embed.device, dtype=padded_embed.dtype)
        )
        hidden_states = (padded_embed + positions)[valid_mask]

        # Attention blocks are n_window_infer wide, which may span several
        # convolution chunks; an item's final block is whatever is left over.
        window_aftercnn = valid_mask.size(-1) * (self.n_window_infer // window)
        if window_aftercnn < 1:
            raise ValueError("n_window_infer must be at least n_window * 2")
        block_lengths: list[int] = []
        for length in aftercnn_lens.tolist():
            block_lengths.extend([window_aftercnn] * (length // window_aftercnn))
            if length % window_aftercnn:
                block_lengths.append(length % window_aftercnn)
        cu_seqlens = torch.tensor(
            [0, *block_lengths], device=hidden_states.device
        ).cumsum(-1, dtype=torch.int32)
        attention_mask = self._block_attention_mask(hidden_states, cu_seqlens)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        return self.proj2(F.gelu(self.proj1(self.ln_post(hidden_states))))


__all__ = [
    "AudioEncoder",
    "AudioEncoderConfig",
    "encoder_frames_from_mel_frames",
    "feature_extract_output_lengths",
]
