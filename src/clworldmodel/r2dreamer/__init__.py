"""Project-owned R2-Dreamer integration for continual ARROW replay studies."""

from .config import R2DreamerConfig

__all__ = [
    "R2DreamerAgent",
    "R2DreamerConfig",
    "R2PolicyState",
    "R2ReplayBatch",
    "R2UpdateResult",
]


def __getattr__(name: str):
    if name in {"R2DreamerAgent", "R2PolicyState", "R2ReplayBatch", "R2UpdateResult"}:
        from .agent import R2DreamerAgent, R2PolicyState, R2ReplayBatch, R2UpdateResult

        return {
            "R2DreamerAgent": R2DreamerAgent,
            "R2PolicyState": R2PolicyState,
            "R2ReplayBatch": R2ReplayBatch,
            "R2UpdateResult": R2UpdateResult,
        }[name]
    raise AttributeError(name)
