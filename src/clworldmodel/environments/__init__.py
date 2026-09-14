# SPDX-License-Identifier: Apache-2.0
"""Project-owned environment boundaries; importing core requires no simulator."""
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PreparedEnvironmentFactory(Protocol):
    """Picklable factory for already preprocessed Gymnasium environments.

    Observations at this boundary are HWC uint8 RGB. The collector alone
    normalizes/reorders them. A factory owns action semantics and preprocessing;
    shared trainers and policies must not inspect environment names.
    """
    action_count: int
    dummy_previous_action: int

    def prepare(self, env_repeat: int, action_seed: int | None) -> Any: ...
