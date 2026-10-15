"""Keep turn-boundary tokens on the greedy path while sampling the content.

Transcribed from ``simulst_infer/vllm_boundary.py`` with one change: the token
ids. That version reads module-level globals that ``SimulstEngine`` assigns
before constructing the LLM (``engine.py:276-277``). Under ``vllm serve`` there
is no engine object to assign them, so they are resolved here from the merged
checkpoint -- ``config.json`` first, then the tokenizer, then the known ids as a
last resort. The behaviour is otherwise identical.

Why the masking exists at all: each assistant turn encodes a
simultaneous-translation policy decision, not just wording.

* empty content + EOT means WAIT (do not translate this chunk yet),
* content + EOT means EMIT,
* EOS is the model's utterance-end prediction.

Sampling a content token trades one wording for another, which is the point of
enabling sampling. Sampling a *boundary* token instead flips the policy on a
coin toss. So this processor decides, per step, which regime applies: if the
argmax is already a boundary token the row is left alone and the boundary wins;
otherwise the boundary tokens are masked to -inf and the sampler chooses among
content tokens only. The boundary ends up being whatever greedy decoding would
have produced, while the wording is sampled.
"""

from __future__ import annotations

import logging

import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor, MoveDirectionality

from simulst_mm.plugin import ARCHITECTURE

logger = logging.getLogger(__name__)

# The literal markers in this tokenizer. TURN_END_TOKEN matches
# hytuner.utils.dataset.audio_sft_dataset.TURN_END_TOKEN; the ids are the ones
# that vocabulary resolves them to, kept only as a last-resort fallback.
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
    """Whether the model being served is the unified speech architecture.

    The ``vllm.logits_processors`` entry point is global: once this package is
    installed, vLLM constructs this processor for *every* model served from the
    environment. Masking ids 120008/120025 in an unrelated vocabulary would be
    silent corruption, so anything that is not our architecture gets a processor
    that does nothing.
    """
    architectures = getattr(vllm_config.model_config.hf_config, "architectures", None) or []
    return ARCHITECTURE in architectures


def resolve_boundary_ids(vllm_config) -> tuple[int, int]:
    """Find (turn_end_token_id, eos_token_id) for the model being served.

    ``merge_export`` writes ``turn_end_token_id`` into config.json, so the first
    branch is the normal one; the tokenizer lookup covers a checkpoint merged by
    an older version, and the constants cover a tokenizer that cannot be loaded
    inside the worker process.
    """
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
        except Exception as error:  # pragma: no cover - depends on the checkpoint
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
    """Mask EOT/EOS for sampling unless they are already the argmax."""

    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool) -> None:
        self.device = device
        self.enabled = serves_this_model(vllm_config)
        if not self.enabled:
            # Installed globally, active only for our architecture. active_rows
            # stays empty, so apply() returns the logits untouched.
            self.boundary_ids = torch.empty(0, dtype=torch.long, device=device)
            self.active_rows: set[int] = set()
            return

        turn_end_id, eos_id = resolve_boundary_ids(vllm_config)
        self.turn_end_token_id = turn_end_id
        self.eos_token_id = eos_id
        self.boundary_ids = torch.tensor(
            [turn_end_id, eos_id], dtype=torch.long, device=device
        )
        # Rows that opted in, i.e. requests that are sampling rather than greedy.
        self.active_rows: set[int] = set()

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        return None

    def is_argmax_invariant(self) -> bool:
        # Masking never removes the argmax: when the argmax is a boundary token
        # the row is skipped entirely, so greedy results are unchanged.
        return True

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None or not self.enabled:
            return

        for index in batch_update.removed:
            self.active_rows.discard(index)

        for index, params, _prompt_ids, _output_ids in batch_update.added:
            # Greedy requests need no protection; argmax already decides.
            sampling = getattr(params, "temperature", 0.0) > 0.0
            if sampling:
                self.active_rows.add(index)
            else:
                self.active_rows.discard(index)

        for first, second, directionality in batch_update.moved:
            first_active = first in self.active_rows
            second_active = second in self.active_rows
            if directionality == MoveDirectionality.SWAP:
                self.active_rows.discard(first)
                self.active_rows.discard(second)
                if second_active:
                    self.active_rows.add(first)
                if first_active:
                    self.active_rows.add(second)
            else:
                # Unidirectional move: first replaces second, first is vacated.
                self.active_rows.discard(second)
                if first_active:
                    self.active_rows.add(second)
                self.active_rows.discard(first)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.active_rows:
            return logits

        rows = torch.tensor(
            sorted(index for index in self.active_rows if index < logits.shape[0]),
            dtype=torch.long,
            device=logits.device,
        )
        if rows.numel() == 0:
            return logits

        selected = logits.index_select(0, rows)
        argmax_ids = selected.argmax(dim=-1)
        boundary_ids = self.boundary_ids.to(logits.device)
        # Rows whose best token is a boundary are policy decisions: let them be.
        argmax_is_boundary = (argmax_ids.unsqueeze(-1) == boundary_ids).any(dim=-1)

        maskable = rows[~argmax_is_boundary]
        if maskable.numel() == 0:
            return logits

        for token_id in boundary_ids.tolist():
            logits[maskable, token_id] = float("-inf")
        return logits


__all__ = [
    "BoundaryGreedyLogitsProcessor",
    "TURN_END_TOKEN",
    "resolve_boundary_ids",
]
