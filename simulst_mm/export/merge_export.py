"""Merge the LLM export and the audio frontend export into one servable directory.

The two trained artefacts are produced separately -- ``export_llm.py`` writes 17
safetensors shards and filters out the 403 ``audio_tower.*`` tensors,
``export_audio.py`` writes those tensors into a single file -- because vLLM's
``HYV3ForCausalLM`` has no idea what an audio tower is. Once the architecture is
``HYV3SpeechForCausalLM`` it does, and the two can live in one directory that
``vllm serve`` takes as-is.

What this writes:

* the 17 LLM shards, **hard-linked** rather than copied. They are 57 GB and
  their tensor names do not change: the model class renames ``model.*`` to
  ``language_model.model.*`` at load time with a ``WeightsMapper``, because
  ``safetensors_weights_iterator`` reads names out of the shard files themselves
  and ignores the index. Rewriting 57 GB to change a prefix would be pure waste.
  A copy is used instead when the two paths are on different filesystems.
* one new shard holding the 403 audio tensors, each prefixed ``audio_tower.``.
* a rewritten ``model.safetensors.index.json``. The index only decides *which
  files* get read (``filter_duplicate_safetensors_files``), so the LLM entries
  are carried over unchanged and the audio entries are appended.
* a ``config.json`` with ``architectures: ["HYV3SpeechForCausalLM"]``, an
  ``audio_config`` block describing the encoder, and ``audio_token_id`` /
  ``turn_end_token_id`` at the top level.
* ``preprocessor_config.json`` copied from the pretrained encoder directory, so
  the Whisper mel extractor resolves from the merged directory alone and the
  deployment no longer needs the separate encoder tree mounted.

Source directories are read-only; everything lands in ``--target``.

    python -m simulst_mm.export.merge_export \\
        --llm   /path/to/hf_export \\
        --audio /path/to/hf_export_audio \\
        --target /path/to/hf_export_unified \\
        --encoder-path /path/to/pretrained_audio_encoder
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

ARCHITECTURE = "HYV3SpeechForCausalLM"
AUDIO_PREFIX = "audio_tower."
AUDIO_SHARD_NAME = "model-audio-tower.safetensors"
INDEX_NAME = "model.safetensors.index.json"

# The pretrained encoder directory, needed only for preprocessor_config.json and
# the encoder shape. Nothing is read from it at serving time once the merge has
# run, and both reads fall back to the built-in values below, so leaving this
# unset only costs a printed warning. Override with --encoder-path, or point
# SIMULST_ENCODER_PATH at it.
ENCODER_PATH = os.environ.get("SIMULST_ENCODER_PATH", "")

# The end-of-turn marker, recorded so boundary.py does not have to re-derive it
# from the tokenizer inside every worker process. Matches
# hytuner.utils.dataset.audio_sft_dataset.TURN_END_TOKEN.
TURN_END_TOKEN = "<｜hy_EOT｜>"

# Encoder shape. The pretrained config nests this under
# thinker_config.audio_config together with a few hundred keys of HF
# boilerplate; only the keys the vendored AudioEncoderConfig reads are carried
# over, and they are verified against the source in `_encoder_shape`.
ENCODER_SHAPE_KEYS = (
    "num_mel_bins",
    "encoder_layers",
    "encoder_attention_heads",
    "encoder_ffn_dim",
    "d_model",
    "max_source_positions",
    "n_window",
    "n_window_infer",
    "conv_chunksize",
    "downsample_hidden_size",
    "output_dim",
)

# Used when the pretrained encoder directory is not reachable.
ENCODER_SHAPE_DEFAULT = {
    "num_mel_bins": 128,
    "encoder_layers": 24,
    "encoder_attention_heads": 16,
    "encoder_ffn_dim": 4096,
    "d_model": 1024,
    "max_source_positions": 1500,
    "n_window": 50,
    "n_window_infer": 800,
    "conv_chunksize": 500,
    "downsample_hidden_size": 480,
    "output_dim": 2048,
}

PREPROCESSOR_DEFAULT = {
    "chunk_length": 30,
    "dither": 0.0,
    "feature_extractor_type": "WhisperFeatureExtractor",
    "feature_size": 128,
    "hop_length": 160,
    "n_fft": 400,
    "n_samples": 480000,
    "nb_max_frames": 3000,
    "padding_side": "right",
    "padding_value": 0.0,
    "return_attention_mask": True,
}

# Everything except the shards and the index: tokenizer, chat template, licence.
PASSTHROUGH_SUFFIXES = (".json", ".jinja", ".txt", ".md")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--llm", required=True, help="LLM export directory (read-only)")
    parser.add_argument("--audio", required=True, help="audio frontend export directory")
    parser.add_argument("--target", required=True, help="output directory")
    parser.add_argument(
        "--encoder-path",
        default=ENCODER_PATH,
        help="pretrained encoder directory, read for preprocessor_config.json and "
        "the encoder shape",
    )
    parser.add_argument(
        "--copy-shards",
        action="store_true",
        help="copy the LLM shards instead of hard-linking them (57 GB)",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing target")
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _encoder_shape(encoder_path: Path | None) -> dict:
    """Pull the encoder shape out of the pretrained config, falling back to defaults.

    A mismatch between what the pretrained config says and what the merged
    config claims would produce an audio tower whose parameter shapes do not
    match the trained tensors, which surfaces much later as an opaque load
    error. Reading the real file keeps the two in step.
    """
    if encoder_path is None:
        print("-- no --encoder-path given, using the built-in encoder shape")
        return dict(ENCODER_SHAPE_DEFAULT)

    config_file = encoder_path / "config.json"
    if not config_file.is_file():
        print(f"-- {config_file} not readable, using the built-in encoder shape")
        return dict(ENCODER_SHAPE_DEFAULT)

    config = _read_json(config_file)
    audio = config.get("thinker_config", config).get("audio_config")
    if not isinstance(audio, dict):
        print(f"-- no thinker_config.audio_config in {config_file}, using defaults")
        return dict(ENCODER_SHAPE_DEFAULT)

    shape = {key: audio[key] for key in ENCODER_SHAPE_KEYS if key in audio}
    missing = [key for key in ENCODER_SHAPE_KEYS if key not in shape]
    for key in missing:
        shape[key] = ENCODER_SHAPE_DEFAULT[key]
    if missing:
        print(f"-- encoder config is missing {missing}, defaulted")

    differing = {
        key: (ENCODER_SHAPE_DEFAULT[key], value)
        for key, value in shape.items()
        if ENCODER_SHAPE_DEFAULT[key] != value
    }
    if differing:
        print(f"-- encoder shape differs from the expected one: {differing}")
    return shape


def _preprocessor_config(encoder_path: Path | None) -> dict:
    if encoder_path is None:
        print("-- no --encoder-path given, writing the built-in preprocessor config")
        return dict(PREPROCESSOR_DEFAULT)

    source = encoder_path / "preprocessor_config.json"
    if not source.is_file():
        print(f"-- {source} not readable, writing the built-in preprocessor config")
        return dict(PREPROCESSOR_DEFAULT)

    config = _read_json(source)
    # processor_class names a Qwen3ASRProcessor that is not installed here and
    # is not needed: only the WhisperFeatureExtractor fields are used.
    config.pop("processor_class", None)
    return config


def _link_or_copy(source: Path, target: Path, *, copy: bool) -> str:
    if target.exists():
        target.unlink()
    if not copy:
        try:
            os.link(source, target)
            return "link"
        except OSError:
            # Different filesystem, or a mount that does not allow links.
            pass
    shutil.copy2(source, target)
    return "copy"


def _turn_end_token_id(llm_dir: Path) -> int | None:
    """Resolve the EOT id from the exported tokenizer, if it is readable."""
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(llm_dir), local_files_only=True)
        encoded = list(tokenizer.encode(TURN_END_TOKEN, add_special_tokens=False))
    except Exception as error:
        print(f"-- could not resolve {TURN_END_TOKEN!r} from the tokenizer: {error}")
        return None

    if len(encoded) != 1:
        print(f"-- {TURN_END_TOKEN!r} encodes to {encoded}, not a single id; skipping")
        return None
    return int(encoded[0])


def main(argv=None) -> int:
    args = parse_args(argv)

    llm_dir = Path(args.llm)
    audio_dir = Path(args.audio)
    target = Path(args.target)
    # Empty means "not supplied": Path("") is Path("."), which would silently
    # read ./config.json from wherever the merge happens to run.
    encoder_path = Path(args.encoder_path) if args.encoder_path else None

    index_path = llm_dir / INDEX_NAME
    audio_weights = audio_dir / "audio_frontend.safetensors"
    audio_manifest_path = audio_dir / "audio_frontend.json"
    for required in (llm_dir / "config.json", index_path, audio_weights, audio_manifest_path):
        if not required.is_file():
            raise SystemExit(f"missing required input: {required}")

    if target.exists():
        if not args.overwrite:
            raise SystemExit(f"{target} already exists (pass --overwrite to replace it)")
        if target.resolve() in (llm_dir.resolve(), audio_dir.resolve()):
            raise SystemExit("refusing to overwrite a source directory")
        print(f"-- removing existing {target}")
        shutil.rmtree(target)
    target.mkdir(parents=True)

    from safetensors.torch import load_file, save_file

    # --- audio tensors -----------------------------------------------------
    audio_manifest = _read_json(audio_manifest_path)
    audio_tensors = load_file(str(audio_weights))
    prefixed = {AUDIO_PREFIX + name: tensor for name, tensor in audio_tensors.items()}
    audio_bytes = sum(tensor.numel() * tensor.element_size() for tensor in prefixed.values())
    save_file(prefixed, str(target / AUDIO_SHARD_NAME), metadata={"format": "pt"})
    print(f"-- wrote {len(prefixed)} audio tensors ({audio_bytes / 1e9:.2f} GB) under {AUDIO_PREFIX}")

    # --- LLM shards --------------------------------------------------------
    index = _read_json(index_path)
    weight_map: dict[str, str] = dict(index["weight_map"])
    shard_names = sorted(set(weight_map.values()))

    mode = None
    for shard in shard_names:
        mode = _link_or_copy(llm_dir / shard, target / shard, copy=args.copy_shards)
    print(f"-- {mode}ed {len(shard_names)} LLM shards ({len(weight_map)} tensors)")

    # --- index -------------------------------------------------------------
    # LLM names stay as they are; HYV3SpeechForCausalLM.hf_to_vllm_mapper does
    # the `model.` -> `language_model.model.` rename when the weights are read.
    for name in prefixed:
        weight_map[name] = AUDIO_SHARD_NAME
    total_size = int(index.get("metadata", {}).get("total_size", 0)) + audio_bytes
    _write_json(target / INDEX_NAME, {"metadata": {"total_size": total_size}, "weight_map": weight_map})

    # --- config ------------------------------------------------------------
    config = _read_json(llm_dir / "config.json")
    config["architectures"] = [ARCHITECTURE]
    config["audio_token_id"] = int(audio_manifest["audio_token_id"])

    turn_end_id = _turn_end_token_id(llm_dir)
    if turn_end_id is not None:
        # Read by boundary.py, which has no tokenizer of its own.
        config["turn_end_token_id"] = turn_end_id

    audio_config = _encoder_shape(encoder_path)
    audio_config.update(
        # The deployed frontend leaves attn_implementation at the transformers
        # default, which on this install is sdpa. Pinned so a future default
        # cannot silently change the numerics.
        attn_implementation="sdpa",
        attention_dropout=0.0,
        encoder_dim=int(audio_manifest["encoder_dim"]),
        fsk_size=int(audio_manifest["fsk_size"]),
        llm_dim=int(audio_manifest["llm_dim"]),
        sample_rate=int(audio_manifest["sample_rate"]),
        hop_length=int(audio_manifest["hop_length"]),
        max_samples=int(audio_manifest["max_samples"]),
    )
    if audio_config["llm_dim"] != config["hidden_size"]:
        raise SystemExit(
            f"llm_dim {audio_config['llm_dim']} does not match the LLM hidden_size "
            f"{config['hidden_size']}; the adapter output would not fit the embedding space"
        )
    if audio_config["encoder_dim"] != audio_config["output_dim"]:
        raise SystemExit(
            f"encoder_dim {audio_config['encoder_dim']} does not match the encoder "
            f"output_dim {audio_config['output_dim']}"
        )
    config["audio_config"] = audio_config
    _write_json(target / "config.json", config)

    _write_json(target / "preprocessor_config.json", _preprocessor_config(encoder_path))

    # --- tokenizer and the rest -------------------------------------------
    carried = []
    for entry in sorted(llm_dir.iterdir()):
        if not entry.is_file() or entry.suffix not in PASSTHROUGH_SUFFIXES:
            continue
        if entry.name in {"config.json", INDEX_NAME}:
            continue
        shutil.copy2(entry, target / entry.name)
        carried.append(entry.name)
    print(f"-- carried over {len(carried)} auxiliary file(s): {', '.join(carried)}")

    _write_json(
        target / "merge_manifest.json",
        {
            "architecture": ARCHITECTURE,
            "llm_source": str(llm_dir.resolve()),
            "audio_source": str(audio_dir.resolve()),
            "encoder_source": str(encoder_path) if encoder_path else None,
            "llm_tensors": len(index["weight_map"]),
            "audio_tensors": len(prefixed),
            "audio_shard": AUDIO_SHARD_NAME,
            "shards_materialised_by": "copy" if args.copy_shards else (mode or "link"),
            "total_size": total_size,
        },
    )

    print(f"-- merged into {target}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
