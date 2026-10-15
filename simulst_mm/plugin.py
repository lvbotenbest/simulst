"""vLLM plugin entry point: register the unified architecture.

vLLM discovers this through the ``vllm.general_plugins`` entry point declared in
pyproject.toml and calls ``register()`` in every process that needs it -- the
engine process (``vllm/v1/engine/core.py``) and each tensor-parallel worker
subprocess (``vllm/v1/worker/worker_base.py``), both before any model is built.
That is why ``vllm serve`` works with no wrapper script and no manual import.

``load_general_plugins`` warns that plugins may be loaded more than once, so
everything here must be idempotent.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

ARCHITECTURE = "HYV3SpeechForCausalLM"
MODEL_PATH = "simulst_mm.model.hy_v3_speech:HYV3SpeechForCausalLM"

_registered = False


def register() -> None:
    """Register the unified speech+LLM architecture with vLLM."""
    global _registered
    if _registered:
        return

    try:
        from vllm import ModelRegistry
    except ImportError as error:  # pragma: no cover - depends on the environment
        # Keep the failure clear for source checkouts and installations made
        # with --no-deps instead of surfacing a bare plugin-hook ImportError.
        raise ImportError(
            "simulst-mm requires vLLM 0.13.x, but vLLM is not installed in "
            "this environment. Install a compatible vLLM before loading the "
            "plugin."
        ) from error

    # Registering lazily by "module:Class" string means importing this plugin
    # does not import torch or hytuner; the class is only resolved when a model
    # with this architecture is actually loaded.
    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCHITECTURE, MODEL_PATH)
        logger.debug("registered %s -> %s", ARCHITECTURE, MODEL_PATH)

    _registered = True
