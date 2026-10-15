"""Serve a speech simultaneous-interpretation stack as a single vLLM model.

The usual design keeps the audio tower outside vLLM: the serving layer loads the
audio encoder and the embedding table itself, splices speech embeddings into
text embeddings, and hands vLLM a finished tensor via ``EmbedsPrompt``. That
works, but it costs a whole GPU for a 330M-parameter tower, it needs two export
directories kept in sync, and it rules out ``vllm serve``.

This package packs both halves into one model so deployment is just:

    vllm serve <unified_dir> --tensor-parallel-size 2 --enable-expert-parallel

Note on what this is *not* for: prefix caching already works on the
EmbedsPrompt path. Measured on a real engine, steady-state cache hit is 0.921
and only 5.5% of prompt tokens are ever prefilled. The compute saved here is
only the ~9.6 ms/turn of re-hashing 12.9 MB of embeddings, about 1.2% of an
800 ms chunk budget. The reason to do this is the deployment shape.
"""

__all__ = ["register"]


def __getattr__(name: str):
    # Keep ``import simulst_mm`` free of torch/vllm imports; the plugin entry
    # point is resolved as ``simulst_mm.plugin:register`` and does not need
    # this, but ``from simulst_mm import register`` should still work.
    if name == "register":
        from simulst_mm.plugin import register

        return register
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
