"""Multi-modal processing: waveforms in, mel features and placeholder ranges out.

One "audio item" is one chunk of the simultaneous-interpretation stream, not one
utterance. Prompts reach this processor in one of two shapes:

* **Pre-expanded:** in-process ``TokensPrompt`` callers provide one complete run
  of ``audio_token_id`` per chunk. The processor locates those runs without
  changing the token ids.
* **One token per chunk:** the OpenAI chat template renders a single
  ``audio_token_id`` for each ``input_audio`` part. The processor expands each
  token to the run required by that chunk.

The shape is determined by counting audio placeholder tokens against the audio
items and their computed speech-token lengths. Both routes end with identical
prompt token ids and placeholder ranges.

Mel extraction runs one waveform at a time. The deployed frontend pads to
``max(len(waveform))`` across the batch, so batching changes the padded mel
width, which in bf16 selects different kernels and shifts embeddings by ~0.03
(``logs/audio_determinism.json``: 12/12 bit-identical singly, 0/12 batched).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from transformers import BatchFeature
from transformers.models.whisper import WhisperFeatureExtractor

from vllm.config.multimodal import BaseDummyOptions
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    MultiModalKwargsOptionalItems,
)
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    MultiModalPromptUpdates,
    PlaceholderFeaturesInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder

from simulst_mm.model.audio_tower import speech_tokens_from_num_samples

# One 800 ms chunk at 16 kHz. Used only to size the profiling run; real chunks
# are whatever the client sends.
_DUMMY_CHUNK_SAMPLES = 12_800


class SpeechProcessingInfo(BaseProcessingInfo):
    """Audio-side configuration, read from the merged checkpoint's config.json."""

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # A live session's prompt grows by one chunk per turn with no fixed
        # bound; the real limit is max_model_len.
        return {"audio": None}

    def get_audio_config(self) -> Mapping[str, Any]:
        config = self.get_hf_config()
        audio_config = getattr(config, "audio_config", None)
        if audio_config is None:
            raise ValueError(
                "config.json has no `audio_config`; the checkpoint was probably not "
                "produced by simulst_mm.export.merge_export"
            )
        if not isinstance(audio_config, Mapping):
            audio_config = audio_config.to_dict()
        return audio_config

    def get_audio_token_id(self) -> int:
        config = self.get_hf_config()
        token_id = getattr(config, "audio_token_id", None)
        if token_id is None:
            raise ValueError("config.json has no `audio_token_id`")
        return int(token_id)

    def get_fsk_size(self) -> int:
        return int(self.get_audio_config().get("fsk_size", 2))

    def get_hop_length(self) -> int:
        return int(self.get_audio_config().get("hop_length", 160))

    def get_max_samples(self) -> int:
        return int(self.get_audio_config().get("max_samples", 480_000))

    def get_sampling_rate(self) -> int:
        return int(self.get_audio_config().get("sample_rate", 16_000))

    def get_feature_extractor(self) -> WhisperFeatureExtractor:
        """Whisper mel extractor, built from the merged directory itself.

        ``merge_export`` copies the encoder's ``preprocessor_config.json`` in, so
        this resolves without the separate encoder directory the deployed
        frontend still needs.
        """
        return cached_feature_extractor(self.ctx.model_config.model)

    def tokens_per_chunk(self, num_samples: int) -> int:
        return speech_tokens_from_num_samples(
            num_samples,
            fsk_size=self.get_fsk_size(),
            max_samples=self.get_max_samples(),
            hop_length=self.get_hop_length(),
        )


@lru_cache(maxsize=4)
def cached_feature_extractor(model_path: str) -> WhisperFeatureExtractor:
    # Cached on a module-level function rather than a method, so the key is the
    # path and not the (unhashable, per-request) info object.
    return WhisperFeatureExtractor.from_pretrained(model_path, local_files_only=True)


def _is_audio_update_applied(found: int, num_items: int, expanded: int) -> bool:
    if found == expanded:
        return True
    if found == num_items:
        return False
    raise ValueError(
        f"prompt has {found} audio placeholder tokens for {num_items} audio chunks; "
        f"expected either one per chunk ({num_items}) or the expanded total "
        f"({expanded})"
    )


class SpeechDummyInputsBuilder(BaseDummyInputsBuilder[SpeechProcessingInfo]):
    """Inputs for the memory-profiling run.

    The dummy text must carry a full run of placeholders per item, not a single
    one, because [`SpeechMultiModalProcessor._get_prompt_updates`][] targets the
    whole run. ``_DUMMY_CHUNK_SAMPLES`` fixes the run length.
    """

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        placeholder = self.info.get_tokenizer().decode([self.info.get_audio_token_id()])
        tokens_per_chunk = self.info.tokens_per_chunk(_DUMMY_CHUNK_SAMPLES)
        return placeholder * (tokens_per_chunk * num_audios)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)
        overrides = mm_options.get("audio") if mm_options else None
        return {
            "audio": self._get_dummy_audios(
                length=_DUMMY_CHUNK_SAMPLES,
                num_audios=num_audios,
                overrides=overrides,
            )
        }


class SpeechMultiModalProcessor(BaseMultiModalProcessor[SpeechProcessingInfo]):
    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=self.info.get_sampling_rate())

    def _mel_length(self, num_samples: int, hop_length: int) -> int:
        """Round the padded waveform up to a whole number of hops.

        Whisper-style extractors derive the frame mask with ``[:, ::hop_length]``
        and drop the final frame when the padded length is not a hop multiple,
        which would leave the encoder one frame short of the token count the
        prompt reserves.
        """
        return -(-num_samples // hop_length) * hop_length

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        # input_ids must be a tensor, not a list: the caller does
        # `processed_data.pop("input_ids").tolist()` (processing.py:1545).
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)

        audios = mm_data.get("audios") or []
        if not audios:
            return BatchFeature(dict(input_ids=input_ids))

        feature_extractor = self.info.get_feature_extractor()
        hop_length = self.info.get_hop_length()
        sampling_rate = self.info.get_sampling_rate()

        features: list[torch.Tensor] = []
        token_lens: list[int] = []
        for audio in audios:
            waveform = np.asarray(audio, dtype=np.float32)
            # One waveform per call: see the module docstring on batching.
            extracted = feature_extractor(
                [waveform],
                sampling_rate=sampling_rate,
                return_attention_mask=True,
                padding="max_length",
                max_length=self._mel_length(len(waveform), hop_length),
                return_tensors="pt",
            )
            length = int(extracted["attention_mask"].sum())
            features.append(extracted["input_features"][0, :, :length].contiguous())
            token_lens.append(self.info.tokens_per_chunk(len(waveform)))

        return BatchFeature(
            dict(
                input_ids=input_ids,
                audio_features=features,
                audio_token_len=torch.tensor(token_lens, dtype=torch.long),
            )
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # One chunk per item, so plain batching along the item axis. Mel widths
        # differ between chunks, which is why audio_features stays a list of
        # tensors rather than a padded stack.
        return dict(
            audio_features=MultiModalFieldConfig.batched("audio"),
            audio_token_len=MultiModalFieldConfig.batched("audio"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        audio_token_id = self.info.get_audio_token_id()
        token_lens = out_mm_kwargs.get_data().get("audio_token_len")

        def placeholder_run(item_idx: int) -> list[int]:
            return [audio_token_id] * int(token_lens[item_idx])

        # The chat template emits one target token per chunk; replacement grows
        # it to the chunk's full run. For pre-expanded prompts only replacement
        # is read by the left-to-right scanner below.
        return [
            PromptReplacement(
                modality="audio",
                target=[audio_token_id],
                replacement=placeholder_run,
            )
        ]

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        prompt_ids: list[int],
        mm_kwargs: MultiModalKwargsOptionalItems,
        mm_prompt_updates: MultiModalPromptUpdates,
        is_update_applied: bool,
    ) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
        """Locate expanded placeholders or expand one-token-per-chunk inputs.

        vLLM's ``is_update_applied`` flag describes whether the prompt originated
        as text or token ids, not which placeholder shape this model received.
        Count the placeholder tokens instead: the expanded form has the sum of
        all chunk run lengths, while the OpenAI form has one per chunk.

        Expanded runs must use vLLM's left-to-right placeholder scanner. Its
        per-item matcher cannot distinguish multiple runs made from the same
        token when their lengths differ.
        """
        audio_token_id = self.info.get_audio_token_id()
        found = sum(token == audio_token_id for token in prompt_ids)
        num_items = mm_items.get_count("audio", strict=False)
        audios = mm_items.get("audio") if num_items else None
        expanded = sum(
            self.info.tokens_per_chunk(audios.get_audio_length(index))
            for index in range(num_items)
        )
        already_expanded = _is_audio_update_applied(found, num_items, expanded)

        return super()._maybe_apply_prompt_updates(
            mm_items=mm_items,
            prompt_ids=prompt_ids,
            mm_kwargs=mm_kwargs,
            mm_prompt_updates=mm_prompt_updates,
            is_update_applied=already_expanded,
        )


__all__ = [
    "SpeechDummyInputsBuilder",
    "SpeechMultiModalProcessor",
    "SpeechProcessingInfo",
]
