"""Keep turn-boundary tokens on the greedy path while sampling content.

Each assistant turn is a simultaneous-translation policy decision: an empty
turn plus EOT means WAIT, content plus EOT means EMIT, and EOS means the model
predicts utterance end. Sampling may change wording, but must not randomly flip
those boundary decisions.

This implementation targets vLLM 0.31 Model Runner V2. Request state is keyed by
persistent request slot and expanded to logits rows through
``LogitsContext.expanded_idx_mapping``, including speculative-decoding rows.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor

from simulst_mm.plugin import ARCHITECTURE

if TYPE_CHECKING:
    import torch
    from vllm.sampling_params import SamplingParams
    from vllm.v1.worker.gpu.sample.logits_processor import (
        LogitsContext,
        LogitsProcRequestState,
    )

logger = logging.getLogger(__name__)

TURN_END_TOKEN = "<｜hy_EOT｜>"
DEFAULT_TURN_END_TOKEN_ID = 120008
DEFAULT_EOS_TOKEN_ID = 120025


def _single_token_id(tokenizer, text: str) -> int | None:
    try:
        encoded = list(tokenizer.encode(text, add_special_tokens=False))
    except Exception:  # pragma: no cover - tokenizer backends vary
        return None
    return int(encoded[0]) if len(encoded) == 1 else None


def serves_this_model(vllm_config) -> bool:
    """Whether the served model is the unified speech architecture."""
    architectures = (
        getattr(vllm_config.model_config.hf_config, "architectures", None) or []
    )
    return ARCHITECTURE in architectures


def resolve_boundary_ids(vllm_config) -> tuple[int, int]:
    """Resolve ``(turn_end_token_id, eos_token_id)`` for the checkpoint."""
    model_config = vllm_config.model_config
    hf_config = model_config.hf_config

    turn_end_id = getattr(hf_config, "turn_end_token_id", None)
    eos_id = getattr(hf_config, "eos_token_id", None)
    if isinstance(eos_id, (list, tuple)):
        eos_id = eos_id[0] if eos_id else None

    if turn_end_id is None or eos_id is None:
        try:
            from vllm.transformers_utils.tokenizer import get_tokenizer

            tokenizer = get_tokenizer(
                model_config.tokenizer,
                tokenizer_mode=model_config.tokenizer_mode,
                trust_remote_code=model_config.trust_remote_code,
                revision=model_config.tokenizer_revision,
            )
        except Exception as error:  # pragma: no cover - checkpoint dependent
            logger.warning("boundary protection could not load the tokenizer: %s", error)
        else:
            if turn_end_id is None:
                turn_end_id = _single_token_id(tokenizer, TURN_END_TOKEN)
            if eos_id is None:
                eos_id = getattr(tokenizer, "eos_token_id", None)

    if turn_end_id is None:
        logger.warning(
            "boundary protection falling back to the built-in EOT id %d; add "
            "`turn_end_token_id` to config.json to make this explicit",
            DEFAULT_TURN_END_TOKEN_ID,
        )
        turn_end_id = DEFAULT_TURN_END_TOKEN_ID
    if eos_id is None:
        eos_id = DEFAULT_EOS_TOKEN_ID

    return int(turn_end_id), int(eos_id)


class BoundaryGreedyLogitsProcessor(LogitsProcessor):
    """Mask EOT/EOS for sampling unless that boundary is already the argmax."""

    def __init__(
        self,
        vllm_config,
        req_states: LogitsProcRequestState,
    ) -> None:
        self.enabled = serves_this_model(vllm_config)
        self.active_slots = None
        if not self.enabled:
            return

        turn_end_id, eos_id = resolve_boundary_ids(vllm_config)
        for name, token_id in (
            ("turn_end_token_id", turn_end_id),
            ("eos_token_id", eos_id),
        ):
            if not 0 <= token_id < req_states.vocab_size:
                raise ValueError(
                    f"{name}={token_id} is outside vocabulary size "
                    f"{req_states.vocab_size}"
                )

        # Heavy model-runner imports stay out of the frontend entry-point load.
        import torch
        from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor

        self.turn_end_token_id = turn_end_id
        self.eos_token_id = eos_id
        self.active_slots = StagedWriteTensor(
            req_states.max_num_reqs,
            dtype=torch.int32,
            device=req_states.device,
        )

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        return None

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        """Initialize a persistent slot and opt in sampling requests only."""
        if not self.enabled:
            return False

        active = getattr(sampling_params, "temperature", 0.0) != 0.0
        # Always overwrite the recycled slot, including the inactive case.
        self.active_slots.stage_write_elem(req_idx, int(active))
        return active

    def apply_staged_writes(self) -> None:
        if self.active_slots is not None:
            self.active_slots.apply_write()

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        if not self.enabled:
            return logits

        # A request can own several logits rows under speculative decoding.
        active_rows = self.active_slots.gpu[ctx.expanded_idx_mapping] != 0
        argmax_ids = logits.argmax(-1)
        argmax_is_boundary = (argmax_ids == self.turn_end_token_id) | (
            argmax_ids == self.eos_token_id
        )
        maskable = active_rows & ~argmax_is_boundary

        logits[maskable, self.turn_end_token_id] = float("-inf")
        logits[maskable, self.eos_token_id] = float("-inf")
        return logits


__all__ = [
    "BoundaryGreedyLogitsProcessor",
    "TURN_END_TOKEN",
    "resolve_boundary_ids",
]
