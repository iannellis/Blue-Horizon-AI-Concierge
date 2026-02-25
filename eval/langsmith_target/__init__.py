"""LangSmith evaluation target package for multi-turn Blue Horizon runs.

Public API:

    run_example()               — Async target function for LangSmith evaluate.
    ensure_orchestration_ready() — Shared OrchestrationManager initializer.
    EvalCaptureCallback         — Callback for capturing routing/tool artifacts.

"""

from blue_horizon.agents.orchestration import OrchestrationManager
from eval.langsmith_target._callback import EvalCaptureCallback
from eval.langsmith_target.target import ensure_orchestration_ready, run_example

__all__ = [
    "EvalCaptureCallback",
    "OrchestrationManager",
    "ensure_orchestration_ready",
    "run_example",
]
