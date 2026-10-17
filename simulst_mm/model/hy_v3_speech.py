"""The unified architecture: audio tower plus the HYV3 language model.

The language model is built through ``init_vllm_registered_model``, so it is the
real in-tree ``HYV3ForCausalLM`` -- tensor parallel, expert parallel and the
fused MoE kernels all behave exactly as they do when that checkpoint is served
on its own. The audio tower is a plain ``nn.Module`` replicated on every rank
(330M parameters, ~660 MB in bf16), which is what the deployed frontend already
does on its dedicated GPU, only now without the dedicated GPU.

vLLM drives it in two steps: ``embed_multimodal`` turns mel features into speech
embeddings, then ``embed_input_ids`` scatters them into the token embeddings at
the placeholder positions. That scatter is the same row-major replacement
``splice_audio_embeddings`` does in the deployed frontend, so the tensor reaching
the first decoder layer is the one the current ``EmbedsPrompt`` path builds by
hand.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import MultiModalEmbeddings, SupportsMultiModal
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

from simulst_mm.chat_template import AUDIO_CHUNK_PLACEHOLDER
from simulst_mm.model.audio_encoder import AudioEncoderConfig
from simulst_mm.model.audio_tower import AudioTower
from simulst_mm.model.processing import (
    SpeechDummyInputsBuilder,
    SpeechMultiModalProcessor,
    SpeechProcessingInfo,
)

from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    init_vllm_registered_model,
    maybe_prefix,
)


@MULTIMODAL_REGISTRY.register_processor(
    SpeechMultiModalProcessor,
    info=SpeechProcessingInfo,
    dummy_inputs=SpeechDummyInputsBuilder,
)
class HYV3SpeechForCausalLM(nn.Module, SupportsMultiModal):
    # The audio tower's weights live under audio_tower.* in the merged
    # checkpoint and are dense; only the language model's names need the
    # stacked/expert mappings, and those stay inside HYV3ForCausalLM.
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # The LLM shards keep their original `model.*` / `lm_head.*` names so the
    # merge can hard-link 57 GB of safetensors instead of rewriting them; only
    # the one small audio shard is new. Weight names come from the shard files
    # themselves (safetensors_weights_iterator), not from the index, so the
    # rename has to happen here. The audio tower's `speech_encoder.*` /
    # `speech_adapter.*` names already match its module tree once prefixed.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("audio"):
            # The OpenAI chat route uses one wrapped audio token per chunk. The
            # processor expands that token to the chunk's required speech-token
            # run; pre-tokenised callers may still supply an expanded run.
            return AUDIO_CHUNK_PLACEHOLDER

        raise ValueError(f"only the audio modality is supported, got {modality!r}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.multimodal_config = vllm_config.model_config.multimodal_config

        audio_config: Mapping = config.audio_config
        if not isinstance(audio_config, Mapping):
            audio_config = audio_config.to_dict()
        self.audio_token_id = int(config.audio_token_id)

        self.audio_tower = AudioTower(
            AudioEncoderConfig.from_dict(dict(audio_config)),
            llm_dim=int(audio_config.get("llm_dim", config.hidden_size)),
            fsk_size=int(audio_config.get("fsk_size", 2)),
        )

        # The architectures override matters: config.architectures still says
        # HYV3SpeechForCausalLM, which would recurse straight back into this
        # class. Pointing the inner model at the plain LLM architecture is what
        # keeps the MoE path intact.
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["HYV3ForCausalLM"],
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model.",
            connector="audio_tower.speech_adapter.",
            tower_model="audio_tower.speech_encoder.",
        )

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def _parse_audio_input(
        self, **kwargs: object
    ) -> tuple[list[torch.Tensor], list[int]] | None:
        """Pull the per-chunk mel features and expected token counts out of kwargs."""
        audio_features = kwargs.pop("audio_features", None)
        audio_token_len = kwargs.pop("audio_token_len", None)
        if audio_features is None:
            return None

        # Chunks of equal length arrive stacked; unequal ones stay a list.
        if isinstance(audio_features, torch.Tensor):
            features = list(audio_features.unbind(0))
        else:
            features = [feature for feature in audio_features]

        token_lens = [int(length) for length in audio_token_len.flatten()]
        if len(token_lens) != len(features):
            raise RuntimeError(
                f"got {len(features)} audio feature tensors but {len(token_lens)} "
                "token lengths"
            )
        return features, token_lens

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        parsed = self._parse_audio_input(**kwargs)
        if parsed is None:
            return []
        features, token_lens = parsed

        parameter = next(self.audio_tower.parameters())
        embeddings: list[torch.Tensor] = []
        for feature, expected in zip(features, token_lens):
            # One chunk per encoder call. Batching would change the padded mel
            # width and move the embeddings in bf16; the measured drift is 0.03,
            # which is enough to change sampled tokens. See processing.py.
            mel = feature.to(device=parameter.device, dtype=parameter.dtype)
            if mel.ndim != 2:
                raise RuntimeError(
                    f"expected a 2D mel tensor per chunk, got {tuple(mel.shape)}"
                )
            feature_lens = torch.tensor(
                [mel.size(-1)], dtype=torch.long, device=mel.device
            )
            chunk_embeddings, _ = self.audio_tower(mel.contiguous(), feature_lens)
            if chunk_embeddings.size(0) != expected:
                raise RuntimeError(
                    f"chunk produced {chunk_embeddings.size(0)} speech embeddings but "
                    f"the prompt reserves {expected} placeholder tokens"
                )
            embeddings.append(chunk_embeddings)

        return embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        # The placeholder is a real reserved token in this vocabulary, so the
        # language model can embed it and no out-of-vocab buffer is needed.
        if multimodal_embeddings is None or is_multimodal is None:
            return super().embed_input_ids(input_ids)

        return super().embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None

        return self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # EPLB and the MoE weight loader both reach for this on the top-level
        # model; forward it to the language model that actually has experts.
        return self.language_model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


__all__ = ["HYV3SpeechForCausalLM"]
