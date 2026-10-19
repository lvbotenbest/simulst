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
from importlib.metadata import PackageNotFoundError, version

logger = logging.getLogger(__name__)

ARCHITECTURE = "HYV3SpeechForCausalLM"
LANGUAGE_ARCHITECTURE = "HYV3ForCausalLM"
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
        raise ImportError(
            "simulst requires upstream vLLM 0.31.x. Install a compatible "
            "server environment with `pip install 'simulst[server]'`, or "
            "install vLLM first and then install simulst with --no-deps."
        ) from error

    try:
        installed_version = version("vllm")
    except PackageNotFoundError as error:  # pragma: no cover - broken environment
        raise RuntimeError("vLLM is importable but has no package metadata") from error
    if not installed_version.split("+", 1)[0].startswith("0.31."):
        raise RuntimeError(
            f"simulst supports upstream vLLM 0.31.x; found {installed_version}"
        )

    supported = set(ModelRegistry.get_supported_archs())
    if LANGUAGE_ARCHITECTURE not in supported:
        raise RuntimeError(
            f"vLLM {installed_version} does not provide {LANGUAGE_ARCHITECTURE}; "
            "install an upstream vLLM 0.31.x build that includes hy_v3"
        )

    # Register lazily by module path only after every compatibility check passes.
    if ARCHITECTURE not in supported:
        ModelRegistry.register_model(ARCHITECTURE, MODEL_PATH)
        logger.debug("registered %s -> %s", ARCHITECTURE, MODEL_PATH)

    _registered = True
