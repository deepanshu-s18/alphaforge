"""Agent runtime context: retries, cost accounting, shared data frames.

LangGraph state carries only JSON-safe metadata (per the spec's Pydantic
ResearchState); heavy DataFrames live in a RunContext closed over by the
node functions for the duration of one orchestrated run.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass, field

import pandas as pd

from alphaforge.state.schema import AgentConfig, AgentError
from alphaforge.utils.logging import get_logger

log = get_logger("alphaforge.agents")


class ToolCallFailed(Exception):
    pass


def with_retries(retries: int = 3, base_delay: float = 0.2):
    """Exponential backoff wrapper; raises ToolCallFailed after the final attempt."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:  # noqa: BLE001 - agents must not crash the graph
                    if attempt == retries:
                        raise ToolCallFailed(
                            f"{fn.__name__} failed after {retries} attempts: {e}"
                        ) from e
                    log.warning("tool_call_retry", fn=fn.__name__, attempt=attempt, error=str(e))
                    time.sleep(delay)
                    delay *= 2

        return wrapper

    return deco


@dataclass
class RunContext:
    """Run-scoped resources shared across agent nodes."""

    config: AgentConfig
    data: dict[str, pd.DataFrame] = field(default_factory=dict)  # "ohlcv"/"earnings" frames
    first_try_ok: int = 0
    tool_calls: int = 0
    errors: list[AgentError] = field(default_factory=list)

    def record_tool(self, ok: bool):
        self.tool_calls += 1
        if ok:
            self.first_try_ok += 1

    @property
    def first_try_rate(self) -> float:
        return self.first_try_ok / self.tool_calls if self.tool_calls else 0.0
