# AWM-AutoRoute mechanism ablations v1 — Atari

## Fixed parent protocol

All four variants inherit `AWM-AutoRoute v4` unchanged: the ARROW original-six
Atari order, 90 epochs per task, seeds S0=`123456789`, S1=`1337`,
S2=`31337`, task-aware training, task-ID-free two-frame inference, replay and
interaction budgets, periodic/final held-out evaluation, private MLP behavior,
and checkpoint/evaluation isolation. Each run sets exactly one typed
`awm_ablation` value; combinations are invalid.

## Single-variable variants

| Variant | Only intended change | Compute accounting |
|---|---|---|
| `no_reuse` | Set `task_mechanism_reuse=false`. New per-task Q/F/P atoms remain, but routes to older atoms are neither allocated nor evaluated. | Same 552,000 WM and 432,000 AC optimizer updates as the parent. The omitted reuse-route parameters are reported rather than padded with inert parameters. |
| `no_functional_protection` | Set posterior, hidden, frozen-actor, shared prediction-head, and compression Q/F/P teacher-distillation weights to zero. Real LTDM Dreamer loss, replay, boundary teachers, consolidation, RCC, and gradient projection remain. | Same optimizer, sampling, and evaluation counts as the parent. Teacher forwards remain in the fixed execution path but have zero functional-target weight. |
| `no_conflict_projection` | Set `component_gradient_projection=false`; current and replay-memory gradients are summed without the conflicting-current-direction projection. | Same 552,000 WM and 432,000 AC optimizer updates as the parent. |
| `no_rcc` | Disable return-gated structural Q/F/P compaction. Dense acquired Q/F/P modules remain dense; ordinary boundary consolidation and rollback remain. | Predeclared non-compute-matched diagnostic: 540,000 online plus 6,000 consolidation WM updates, no 6,000 RCC candidate updates and no 480 RCC-only selection rollouts. Reported periodic/final evaluation is unchanged. |

No variant changes environment frames, online update counts, replay capacity or
sampling, task duration/order, action handling, observation/reward processing,
task-label exposure, or reported evaluation opportunities. `no_rcc` is the
sole declared exception for removed mechanism-internal candidate compute; it
must not be presented as compute matched.

## Launch and evidence contract

Each cell is a fresh run from one clean pushed commit and records the exact
variant, resolved config, protocol string, budget ledger, dependency/runtime
state, seed, output path, and project commit. Smoke tests must validate config,
one WM/AC update, routing, and checkpoint topology before the campaign starts.
Failed/stopped attempts remain evidence and never silently fill a cell.

```bash
python scripts/run_evolving_atomic_rssm_d_autoroute.py \
  --seed 0 --awm-ablation no_reuse --dry-run
```

