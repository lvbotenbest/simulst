# simulst-mm

Serve a speech simultaneous-interpretation stack — audio encoder, adapter, and
LLM — as a **single vLLM model**.

```bash
pip install simulst-mm
vllm serve /path/to/unified_checkpoint --tensor-parallel-size 2 --enable-expert-parallel
```

No wrapper script, no manual `import`, no second model directory. Installing the
package is enough: it registers the architecture through vLLM's
`vllm.general_plugins` entry point, which vLLM calls in the engine process and
in every tensor-parallel worker before the model is built.

## What problem this solves

The usual way to serve a streaming speech-to-text model on vLLM is to keep the
audio tower *outside* the engine: your serving layer loads the encoder and the
embedding table, splices speech embeddings into text embeddings, and hands vLLM
a finished tensor via `EmbedsPrompt`.

That works, but it costs:

- **a whole GPU** for a 330M-parameter tower sitting next to a tensor-parallel LLM;
- **two export directories** that must be kept in sync;
- **`vllm serve` itself** — you need a custom engine script instead.

This package merges both halves into one checkpoint and one model class, so the
tower is sharded onto the same GPUs as the LLM and deployment is a stock
`vllm serve` command.

### What it is *not* for

Prefix caching. That already works on the `EmbedsPrompt` path — measured on a
real engine, steady-state cache hit is 0.921 and only 5.5% of prompt tokens are
ever prefilled. The compute saved here is about 9.6 ms/turn of re-hashing
embeddings, roughly 1.2% of an 800 ms chunk budget. **The reason to do this is
the deployment shape.**

## Install

```bash
pip install simulst-mm
```

The package requires **vLLM 0.13.x** and declares `vllm>=0.13,<0.14`
(developed and verified against 0.13.0). PyTorch, transformers, and safetensors
are installed through vLLM's dependency set.

If the target environment already provides a compatible custom or locally built
vLLM, use `pip install --no-deps simulst-mm` to prevent pip from replacing it.

Verify the install:

```bash
python -c "
from importlib.metadata import entry_points
for g in ('vllm.general_plugins', 'vllm.logits_processors'):
    print(g, {e.name: e.value for e in entry_points(group=g)})
"
```

You should see `simulst_mm` and `simulst_boundary`.

## Preparing a checkpoint

The package expects one directory containing the LLM weights, the audio tower
weights under an `audio_tower.` prefix, and a `config.json` whose
`architectures` is `["HYV3SpeechForCausalLM"]`. If you have the two halves
exported separately, merge them:

```bash
python -m simulst_mm.export.merge_export \
  --llm    /path/to/hf_export \
  --audio  /path/to/hf_export_audio \
  --target /path/to/unified_checkpoint \
  --encoder-path /path/to/pretrained_audio_encoder   # optional
```

The LLM shards are **hard-linked**, not copied — tensor names are rewritten at
load time by a `WeightsMapper`, so there is no reason to rewrite tens of GB on
disk. A copy is used automatically when the paths are on different filesystems.

`--encoder-path` is only read for `preprocessor_config.json` and the encoder
shape, and falls back to built-in defaults when absent. Nothing outside the
merged directory is read at serving time.

## Usage

Prompts arrive **pre-tokenised**, with one run of audio placeholder tokens per
chunk already in place — the simultaneous-interpretation protocol (turn
structure, WAIT/EMIT policy, rolling window) stays in your client, where it
belongs.

```python
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

llm = LLM(
    model="/path/to/unified_checkpoint",
    tensor_parallel_size=2,
    enable_expert_parallel=True,
    dtype="bfloat16",
    enforce_eager=True,
)

prompt = TokensPrompt(
    prompt_token_ids=token_ids,                        # placeholders already present
    multi_modal_data={"audio": [chunk_waveform, ...]},  # one entry per chunk
)
llm.generate([prompt], SamplingParams(temperature=0.7, max_tokens=128))
```

### Known limitation: audio over HTTP

`/v1/completions` **cannot carry audio**. `CompletionRequest` has
`prompt_embeds` but no multi-modal field, so a `multi_modal_data` key is
silently dropped — the server logs
`fields ... ignored: {'multi_modal_data'}` and answers from the text prompt
alone.

Audio over HTTP would have to go through `/v1/chat/completions` with
`input_audio` content parts, which inserts placeholders via
`get_placeholder_str` and the chat template. This model's placeholders are
produced upstream instead, so that route is **not yet implemented**. Until it
is, drive the model in-process with `TokensPrompt` as shown above.

## Design notes

**One chunk per encoder call.** Whisper-style feature extraction pads to the
longest waveform in the batch, so batching changes the padded mel width, which
in bf16 selects different kernels and shifts embeddings by ~0.03. Encoding one
chunk at a time is bit-exact across repeated calls (12/12 identical); batching
is not (0/12). The processor therefore extracts one waveform per call.

**Placeholders are located, not generated.** vLLM's default per-item matcher
cannot handle several placeholder runs made of the *same* token id: a shorter
later run is a prefix of an earlier longer one, matches first, and the
one-non-empty-REPLACE-per-pass rule then strands the earlier item. The
processor overrides `_maybe_apply_prompt_updates` to use vLLM's left-to-right
scanner, which assigns the n-th run to the n-th item and leaves token ids
untouched.

**Boundary protection travels with the model.** A logits processor registered
through `vllm.logits_processors` masks end-of-turn and EOS for *sampling* rows
unless they are already the argmax, so greedy decoding is provably unaffected.
That entry-point group is global — vLLM builds the class for every model served
from the environment — so the processor inspects the architecture and disables
itself for anything else.

## Verification

Parity against a reference deployment that keeps the tower outside vLLM:

| check | result |
|---|---|
| audio embeddings, bit-level | 12/12 identical, max abs diff 0 |
| generated token ids, 8 turns | identical on every turn |
| prefix cache, 40 turns | 0.9432 steady (0.977 by turn 40) |
| boundary processor | 5/5 checks pass |

## License

See `LICENSE`.
