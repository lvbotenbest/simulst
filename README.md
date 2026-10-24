# simulst

Serve a speech simultaneous-interpretation stack — audio encoder, adapter, and
LLM — as a **single vLLM model**.

```bash
# Install the plugin from PyPI.
pip install "simulst==0.1.0"

# Or install vLLM and its audio dependencies in a fresh Python 3.12 environment.
pip install "simulst[server]==0.1.0"

vllm serve /path/to/unified_checkpoint --served-model-name simulst
```

No wrapper script, no manual `import`, no second model directory. Installing the
package is enough: it registers the architecture through vLLM's
`vllm.general_plugins` entry point, which vLLM calls in the engine process and
in every tensor-parallel worker before the model is built. One server can accept
multiple concurrent HTTP requests; vLLM queues and schedules them.

## Install

The plugin supports CPython **3.10, 3.11, 3.12, 3.13, and 3.14**
(`>=3.10,<3.15`) and requires upstream **vLLM >=0.31**. Python 3.12 is the
recommended default for a new environment. Choose the installation mode that
matches the environment:

```bash
# Existing vLLM >=0.31 GPU environment.
pip install "simulst==0.1.0"
# Add HTTP audio decoders if the existing vLLM lacks its audio extra.
pip install "simulst[audio]==0.1.0"

# Fresh environment: install vLLM and audio dependencies too.
pip install "simulst[server]==0.1.0"

# Optional checkpoint-export and reference-client tools.
pip install "simulst[export,client]==0.1.0"
```

The base package intentionally does not depend on vLLM. The `audio` extra
installs only the decoder libraries used by vLLM's HTTP `input_audio` path;
it does not depend on vLLM. The `server` extra installs
`vllm[audio]>=0.31`, including the same decoders. The missing upper bound avoids
downgrading a newer vLLM already selected by the operator. It is not a promise
that every future internal vLLM API will remain compatible. vLLM 0.31
currently resolves a substantially newer stack (including torch 2.13 and
Transformers 5) and defaults to CUDA 13 packages, with separate CUDA 12.9
artifacts available. Do not install the `server` extra into an established GPU
environment unless replacing that stack is intended.

Verify the install:

```bash
python -c "
from importlib.metadata import entry_points
for g in ('vllm.general_plugins', 'vllm.logits_processors'):
    print(g, {e.name: e.value for e in entry_points(group=g)})
"
```

You should see `simulst_mm` and `simulst_boundary`.

## Environment compatibility and troubleshooting

`simulst` is pure Python, but it integrates with internal vLLM model,
multimodal-processing, and logits-processing APIs. The supported server stack is
therefore deliberately narrow:

| component | supported / expected |
|---|---|
| Python | `>=3.10,<3.15` |
| vLLM | `>=0.31` (0.31 adapted; 0.32 key interfaces source-checked) |
| PyTorch | 2.13.0 for the vLLM 0.31.0 PyPI wheel |
| Transformers | `>=5.10.4,<5.18` through vLLM |
| FlashInfer core | `flashinfer-python==0.7.0.post1` through vLLM 0.31.0 |
| `flashinfer-cubin` | optional; if installed, its version must exactly match FlashInfer |
| `flashinfer-jit-cache` | optional; its version and CUDA suffix must match FlashInfer |
| external `flash-attn` | not required by `simulst`; an old build may conflict with PyTorch 2.13 |
| `deep-ep` | optional; only needed when that expert-parallel backend is selected |

The vLLM 0.31 implementation and the key vLLM 0.32 AsyncLLM, multimodal, and
Model Runner V2 interfaces have been checked. This is an API compatibility
check, not a claim that every future `>=0.31` release has passed GPU/checkpoint
end-to-end validation. Missing required model architecture or API capabilities
fail at startup and require a `simulst` compatibility update.

The exact torch, Transformers, FlashInfer, and CUDA packages are owned by vLLM.
They are intentionally not pinned a second time by `simulst`. For an existing
GPU environment, install this package with `--no-deps`. For a new environment,
prefer a fresh Python 3.12 virtual environment and install `simulst[server]`.

Inspect the active environment before troubleshooting:

```bash
python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
import torch

for name in (
    "simulst",
    "vllm",
    "torch",
    "transformers",
    "flashinfer-python",
    "flashinfer-cubin",
    "flashinfer-jit-cache",
    "flash-attn",
    "deep-ep",
):
    try:
        print(f"{name}: {version(name)}")
    except PackageNotFoundError:
        print(f"{name}: not installed")
print("torch CUDA:", torch.version.cuda)
PY
```

### `undefined symbol` from `flash_attn_2_cuda` or `deep_ep_cpp`

This means a compiled extension was built against a different PyTorch ABI,
usually because vLLM upgraded torch while an old extension remained installed.
Neither external `flash-attn` nor `deep-ep` is required for a single-GPU default
deployment. If those backends are not intentionally used, remove the stale
extensions and restart vLLM:

```bash
python -m pip uninstall -y flash-attn deep-ep
```

If either backend is required, rebuild it against the exact torch and CUDA
versions in the active environment instead of bypassing the import error.

### FlashInfer package version mismatch

An error such as:

```text
flashinfer-cubin version (X) does not match flashinfer version (Y)
```

means an optional precompiled artifact does not match `flashinfer-python`.
`flashinfer-cubin` and `flashinfer-jit-cache` are not required: FlashInfer can
compile or download kernels on first use. The simplest repair is:

```bash
python -m pip uninstall -y flashinfer-cubin flashinfer-jit-cache
```

If precompiled/offline kernels are required, install matching artifacts from the
FlashInfer wheel index. All FlashInfer package versions must match exactly, and
the JIT-cache index must match the torch CUDA build (`cu129` or `cu130`). Use
`FLASHINFER_DISABLE_VERSION_CHECK=1` only while running FlashInfer's artifact
installation command to replace a mismatched package; do not keep it in the
vLLM service environment.

### GPU memory utilization check fails

If vLLM reports that free memory is below its requested utilization, first stop
other GPU processes or lower the target, for example:

```bash
vllm serve /path/to/unified_checkpoint \
  --served-model-name simulst \
  --gpu-memory-utilization 0.88
```

### Network interface errors from Gloo or NCCL

Single-node vLLM still initializes distributed communication. If an inherited
`GLOO_SOCKET_IFNAME` or `NCCL_SOCKET_IFNAME` names a missing interface, select a
real interface shown by `ip -br addr`. `lo` is suitable only for single-node
runs; multi-node deployments need a routable interface shared by all nodes.

## Preparing a checkpoint

The package expects one directory containing the LLM weights, the audio tower
weights under an `audio_tower.` prefix, and a `config.json` whose
`architectures` is `["HYV3SpeechForCausalLM"]`. If you have the two halves
exported separately, use an existing PyTorch/Transformers export environment,
install the safetensors helper, and merge them:

```bash
pip install "simulst[export]==0.1.0"
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

The merge also installs an audio-aware `chat_template.jinja` and a
`generation_config.json` for `/v1/chat/completions`. Pass
`--no-openai-assets` only if the checkpoint must retain its original template.
An existing merged checkpoint can be updated in place with:

```bash
python -m simulst_mm.chat_template --model /path/to/unified_checkpoint
```

## OpenAI-compatible HTTP deployment and concurrency

Activate the same Python environment that contains vLLM and `simulst`, then
start one vLLM server with the unified checkpoint:

```bash
vllm serve /path/to/unified_checkpoint \
  --served-model-name simulst \
  --host 0.0.0.0 \
  --port 8000
```

That command already supports concurrent HTTP requests. Multiple clients send
independent requests to `/v1/chat/completions`; the one vLLM engine queues them
and applies continuous batching. Do not start one model process per request, and
do not add `AsyncTurnRunner` around the HTTP server.

For example, a two-GPU deployment with capacity for up to eight scheduled
sequences can be started as follows when those settings match the checkpoint and
hardware:

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve /path/to/unified_checkpoint \
  --served-model-name simulst \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --max-num-seqs 8 \
  --host 0.0.0.0 \
  --port 8000
```

`--tensor-parallel-size` and `--enable-expert-parallel` configure model
parallelism, not HTTP concurrency. `--max-num-seqs` is scheduler capacity, not a
hard HTTP connection limit; excess requests may wait. Available admission and
scheduler flags vary by vLLM release, so inspect the installed CLI before adding
version-specific tuning:

```bash
vllm serve --help | grep -E "max-num-seqs|max-num-active-seqs"
```

To run the repository's reference streaming client, use a source checkout
(`examples/` is not installed by the wheel):

```bash
pip install "simulst[client]==0.1.0"
# Refresh an older unified checkpoint once to install the current chat template.
python -m simulst_mm.chat_template --model /path/to/unified_checkpoint
# Default: source text plus translation; appends control token 120808.
python examples/openai_client.py --audio talk.wav --direction en-zh
# Translation-only model: do not append the control token.
python examples/openai_client.py --audio talk.wav --direction en-zh \
  --output-mode translation
```

The example writes each raw server JSON response to stdout. It is only a
protocol reference; production conversation and result handling belong in the
downstream inference service. `/v1/completions` cannot carry audio and is not
supported for this path.

## In-process usage

Pre-tokenised callers may still provide one expanded run of audio placeholder
tokens per chunk. The simultaneous-interpretation protocol (turn structure,
WAIT/EMIT policy, rolling window) remains in the client.

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

## In-process AsyncLLM concurrency without HTTP

This helper is for applications that create vLLM `AsyncLLM` directly instead
of running `vllm serve`. HTTP clients do not need it: send concurrent requests
to the server as described above.

`AsyncTurnRunner` provides a synchronous one-input/one-output facade over one
shared `AsyncLLM`. Each caller blocks for its own final output, while calls from
different threads run concurrently and may finish out of submission order:

```python
from concurrent.futures import ThreadPoolExecutor

from simulst_mm.serving import AsyncTurnRunner

# The factory runs on the runner's private asyncio thread and must return a
# configured vLLM AsyncLLM instance.
runner = AsyncTurnRunner(make_async_llm, timeout_seconds=900)
try:
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(runner.submit, prompt, sampling_params, owner=session_id)
            for session_id, prompt in independent_turns
        ]
        outputs = [future.result() for future in futures]
finally:
    runner.close()
```

Prompts and audio arrays are treated as immutable for the duration of a request.
Owner cancellation affects only that owner's active requests. This helper does
not implement conversation history, HTTP/SSE/WebSocket transport, or
multi-instance routing; those belong in a higher-level serving application.

## Design notes

**One chunk per encoder call.** Whisper-style feature extraction pads to the
longest waveform in the batch, so batching changes the padded mel width, which
in bf16 selects different kernels and shifts embeddings by ~0.03. Encoding one
chunk at a time is bit-exact across repeated calls (12/12 identical); batching
is not (0/12). The processor therefore extracts one waveform per call.

**Two placeholder shapes are supported.** The OpenAI chat template emits one
audio token per chunk and the processor expands it to the required run. Existing
`TokensPrompt` callers may supply those runs pre-expanded; the processor detects
that shape and uses vLLM's left-to-right scanner so runs of the same token remain
associated with the correct chunks.

**Boundary protection travels with the model.** A vLLM 0.31 Model Runner V2
logits processor masks end-of-turn and EOS for sampling requests unless the
boundary is already the argmax, so greedy boundary decisions are preserved.
State is keyed by persistent request slot and expanded through vLLM's row mapping,
including speculative-decoding rows. The entry point is global, so the processor
inspects the architecture and disables itself for every other model.

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
