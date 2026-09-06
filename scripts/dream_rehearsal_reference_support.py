"""Audited boundary to the two unmodified, pinned reference source subsets.

This script-layer integration is the ONLY place that uses their private APIs.
It is intentionally independent from the ARROW model/trainer and never monkey
patches a learning method, optimizer, reward head, or replay sampler.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DR_ROOT = ROOT / "third_party/dream_rehearsal"
NM_ROOT = ROOT / "third_party/nm512_dreamerv3"
SOURCE_PINS = {
    "dream_rehearsal": "7680778f798be3a27a17c320cc875b573c45f0e1",
    "nm512_dreamerv3": "6ef8646d807cd10ce0c88e10a7e943211e7fc44c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_reference_sources() -> dict[str, Any]:
    result = {}
    for name, commit in SOURCE_PINS.items():
        root = ROOT / "third_party" / name
        provenance = json.loads((root / "provenance.json").read_text())
        if provenance["commit"] != commit or provenance["local_modifications"]:
            raise ValueError(f"Reference pin or no-modification contract changed: {name}")
        manifest = {}
        for line in (root / "MANIFEST.sha256").read_text().splitlines():
            expected, relative = line.split("  ", 1)
            if not (root / relative).resolve().is_relative_to(root.resolve()):
                raise ValueError("Source manifest path escapes its vendor root")
            if sha256(root / relative) != expected:
                raise ValueError(f"Reference source differs from pinned artifact: {name}/{relative}")
            manifest[relative] = expected
        if manifest != provenance["files"]:
            raise ValueError(f"Reference manifests disagree: {name}")
        result[name] = {
            "commit": commit, "repository": provenance["repository"],
            "files": manifest, "modified": False,
        }
    return result


def load_reference_runtime():
    """Import explicit modules, refusing an already-loaded ARROW name collision."""
    verify_reference_sources()
    expected_roots = {
        **dict.fromkeys(("dreamer", "models", "networks", "tools", "parallel", "exploration", "envs.wrappers"), NM_ROOT),
        **dict.fromkeys(("orchestrator_chain_nm512", "orchestrator_ab_nm512", "nm512_margin_probe"), DR_ROOT),
    }
    for name, root in expected_roots.items():
        existing = sys.modules.get(name)
        if existing is not None and not Path(existing.__file__).resolve().is_relative_to(root):
            raise RuntimeError(f"Run the reference trainer in a fresh process; module collision: {name}")
    sys.path[:0] = [str(NM_ROOT), str(DR_ROOT / "src")]
    dreamer = importlib.import_module("dreamer")
    tools = importlib.import_module("tools")
    reference = importlib.import_module("orchestrator_chain_nm512")
    parallel = importlib.import_module("parallel")
    wrappers = importlib.import_module("envs.wrappers")
    for name, root in expected_roots.items():
        loaded = sys.modules.get(name)
        if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(root):
            raise RuntimeError(f"Reference import resolved outside its pinned source: {name}")
    return dreamer, tools, reference, parallel, wrappers


def resolve_model_config(protocol, output_dir: Path, *, tools_module=None) -> argparse.Namespace:
    """Compose exactly NM512 defaults + the author's published substrate preset.

    The substrate uses a ``value`` key although NM512 consumes ``critic``; we
    preserve that unused key rather than silently converting it into a tuning
    change. Both the consumed critic settings and the unused key are saved.
    """
    import ruamel.yaml as yaml

    source = yaml.safe_load((NM_ROOT / "configs.yaml").read_text())
    setup = (DR_ROOT / "substrate/SETUP.md").read_text()
    snippet = setup.split("```yaml\n", 1)[1].split("```", 1)[0]
    preset = yaml.safe_load(snippet)["minigrid"]
    resolved = copy.deepcopy(source["defaults"])

    def merge(target, updates):
        for key, value in updates.items():
            if isinstance(value, dict) and key in target:
                merge(target[key], value)
            else:
                target[key] = copy.deepcopy(value)
    merge(resolved, preset)
    if tools_module is not None:
        resolved = {k: tools_module.args_type(v)(v) for k, v in resolved.items()}
    # Necessary Atari/protocol adaptations only. No learning math is changed.
    resolved.update(
        task=protocol.tasks[0], logdir=str(output_dir),
        traindir=str(output_dir / "train_eps"), evaldir=str(output_dir / "eval_eps"),
        seed=protocol.seed, device=protocol.device, compile=False, precision=32,
        video_pred_log=False, expl_behavior="greedy", expl_until=0,
        expl_extr_scale=1.0, num_actions=18, envs=1,
        size=[protocol.image_size, protocol.image_size],
        action_repeat=protocol.action_repeat,
        time_limit=protocol.episode_limit_raw_frames // protocol.action_repeat,
        prefill=protocol.prefill_decisions, eval_episode_num=protocol.eval_episodes,
        steps=int(1e9), eval_every=protocol.chunk_decisions,
        # In NM512 erase_over_episodes, zero means no erasure. With Atari's
        # larger budget the original defaults' one-million cap is NOT enough.
        dataset_size=0,
    )
    expected = {
        "batch_size": protocol.batch_size, "batch_length": protocol.batch_length,
        "imag_horizon": protocol.imag_horizon, "train_ratio": protocol.train_ratio,
        "pretrain": protocol.pretrain_updates,
    }
    for name, value in expected.items():
        if resolved[name] != value:
            raise ValueError(f"Official substrate configuration mismatch: {name}")
    if resolved["actor"]["unimix_ratio"] != 0.01 or resolved["actor"]["lr"] != 3e-5:
        raise ValueError("Official actor distribution/optimizer changed")
    return argparse.Namespace(**resolved)
