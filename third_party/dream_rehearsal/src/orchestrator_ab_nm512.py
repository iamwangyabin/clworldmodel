"""Stage A->B LOCATE RUN (v1.0) orchestrator on the NM512/dreamerv3-torch trainer.

Implements docs/STAGE_AB_LOCATE_PREREG_NM512.md (LOCKED 2026-06-21):
  Phase A : train a FRESH agent on task A (DoorKey-9x9) to stable competence (eval >= 0.6
            over the last 3 evals).  Record A_before. Snapshot the FIXED held S / a* / A-batch
            with MarginLogger (captured while the A-replay still exists).
  Switch  : CLEAR the replay buffer (fresh eps dict + traindir for B) -> B-only replay, the
            naive sequential fine-tuning baseline (pre-reg s4, signed off).
  Phase B : continue the SAME agent on task B (SimpleCrossing). Every eval interval measure,
            at the frozen A-snapshot:  A retention (real-env A eval), margin, flip-fraction,
            WM-on-A recon.  Record A_after.

Everything reuses NM512 components (dreamer.make_env / Dreamer / tools.simulate) so the training
dynamics are byte-identical to a normal `python dreamer.py` run -- we only add the task switch,
the buffer clear, and the per-eval MarginLogger call.

Outputs <logdir>/ab_metrics.jsonl (one row per B-eval) + <logdir>/ab_summary.json (Q1/Q2 read).
This is ONE seed; run n>=3 seeds (per pre-reg) and aggregate across seed dirs.

SMOKE (CPU, tiny budgets, asserts the wiring end-to-end without learning):
  python orchestrator_ab_nm512.py --smoke --logdir ~/dv3_logs/ab_smoke --device cpu
REAL (GPU, one seed):
  python orchestrator_ab_nm512.py --logdir ~/dv3_logs/ab_s1 --seed 1 --device cuda \
      --task_a minigrid_DoorKey-9x9 --task_b minigrid_SimpleCrossingS9N1
"""
from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import pathlib
import sys

# Make the trainer importable whether this file is run from ~/projects/dreamerv3-torch or
# scp'd alongside it (mirrors nm512_margin_probe.py's path bootstrap).
_HERE = pathlib.Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parent, _HERE.parent.parent):
    if (_cand / "dreamer.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
        break


# --------------------------------------------------------------------------------------
# Logger that taps eval_return / train_return as simulate writes them (simulate only reports
# via logger; it returns no scores). We read .captured right after each simulate call.
# --------------------------------------------------------------------------------------
def _make_capture_logger(tools, logdir, step):
    class CaptureLogger(tools.Logger):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.captured = {}

        def scalar(self, name, value):
            super().scalar(name, value)
            self.captured[name] = float(value)

    return CaptureLogger(pathlib.Path(logdir).expanduser(), step)


# --------------------------------------------------------------------------------------
# Config (same stack dreamer.py builds) + small overrides for the orchestrator
# --------------------------------------------------------------------------------------
def _recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            _recursive_update(base[key], value)
        else:
            base[key] = value


def build_config(config_names, overrides):
    import ruamel.yaml as yaml
    import tools

    cfg_path = pathlib.Path(sys.modules["tools"].__file__).parent / "configs.yaml"
    configs = yaml.safe_load(cfg_path.read_text())
    defaults = {}
    for name in ["defaults", *config_names]:
        _recursive_update(defaults, configs[name])

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        parser.add_argument(f"--{key}", type=tools.args_type(value), default=tools.args_type(value)(value))
    config = parser.parse_args([])
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


# --------------------------------------------------------------------------------------
# Env / agent / prefill / eval helpers (thin wrappers over dreamer.py + tools)
# --------------------------------------------------------------------------------------
def build_task_envs(D, config, task, n_train=1, n_eval=1):
    from parallel import Damy

    config.task = task  # make_env reads config.task; spaces are identical across our minigrid tasks
    train = [Damy(D.make_env(config, "train", i)) for i in range(n_train)]
    evals = [Damy(D.make_env(config, "eval", 100 + i)) for i in range(n_eval)]
    return train, evals


def random_prefill(tools, config, train_envs, train_eps, traindir, logger, steps):
    import torch

    n = len(train_envs)
    random_actor = tools.OneHotDist(torch.zeros(config.num_actions).repeat(n, 1))

    def random_agent(o, d, s):
        action = random_actor.sample()
        return {"action": action, "logprob": random_actor.log_prob(action)}, None

    tools.simulate(random_agent, train_envs, train_eps, traindir, logger,
                   limit=config.dataset_size, steps=steps)


def eval_return(tools, agent, eval_envs, eval_eps, evaldir, logger, episodes):
    """Run `episodes` deterministic (training=False) eval episodes; return mean real-env return."""
    eval_policy = functools.partial(agent, training=False)
    logger.captured.pop("eval_return", None)
    tools.simulate(eval_policy, eval_envs, eval_eps, evaldir, logger,
                   is_eval=True, episodes=episodes)
    return logger.captured.get("eval_return")


def wm_rep_fingerprint(wm):
    """SHA-256 over encoder+dynamics parameter bytes -- the frozen_rep structural gate.

    Computed at the A->B switch and again at run end; under --wm_mode frozen_rep the two MUST be
    identical (any drift = the freeze is leaking, run invalid). Cheap (~20M params, CPU copy once)."""
    h = hashlib.sha256()
    for prefix, mod in (("enc", wm.encoder), ("dyn", wm.dynamics)):
        for n, p in sorted(mod.named_parameters(), key=lambda kv: kv[0]):
            h.update(f"{prefix}.{n}".encode())
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def eval_return_with_actor(tools, agent, frozen_actor, eval_envs, eval_eps, evaldir, logger, episodes):
    """A-eval with a specific (frozen) actor swapped into task_behavior, restored in `finally`.

    frozen_actor=None -> ordinary eval_return (the `joint` baseline path takes this; structurally
    identical to before). Otherwise the frozen A-actor reads through the LIVE (B-drifted) world model
    -- isolating pure representation overwrite once actor drift is removed (frozen-head-A arm).
    The finally-restore guarantees an exception mid-eval can NEVER leave the frozen actor live for
    subsequent B-training (pre-reg gate 5.3)."""
    if frozen_actor is None:
        return eval_return(tools, agent, eval_envs, eval_eps, evaldir, logger, episodes)
    tb = agent._task_behavior
    live_actor = tb.actor
    try:
        tb.actor = frozen_actor
        return eval_return(tools, agent, eval_envs, eval_eps, evaldir, logger, episodes)
    finally:
        tb.actor = live_actor


def make_mixed_dataset(D, tools, a_eps, b_eps, config, replay_fraction):
    """B-phase training dataset with task-A rehearsal mixed in at `replay_fraction`.

    f=0 -> delegates to the UNCHANGED make_dataset(b_eps) so it is byte-identical to the no-replay
    baseline (the f=0==baseline gate is structural, not just asserted). f>0 -> every batch has EXACTLY
    n_a = round(f*batch_size) task-A sequences + (batch_size - n_a) task-B sequences (single-variable:
    only the A/B mix changes; LR/budget/etc untouched). Returns (dataset, effective_fraction, n_a)."""
    import numpy as np

    if replay_fraction <= 0:
        return D.make_dataset(b_eps, config), 0.0, 0

    n_a = int(round(replay_fraction * config.batch_size))
    n_a = max(1, min(config.batch_size - 1, n_a))  # keep both tasks present
    n_b = config.batch_size - n_a
    eff = n_a / config.batch_size

    def gen():
        a_s = tools.sample_episodes(a_eps, config.batch_length, seed=0)
        b_s = tools.sample_episodes(b_eps, config.batch_length, seed=1)
        while True:
            seqs = [next(a_s) for _ in range(n_a)] + [next(b_s) for _ in range(n_b)]
            assert len(seqs) == config.batch_size
            yield {k: np.stack([s[k] for s in seqs], 0) for k in seqs[0]}

    return gen(), eff, n_a


# --------------------------------------------------------------------------------------
# The locate run
# --------------------------------------------------------------------------------------
def run(args):
    from collections import OrderedDict
    import torch
    import numpy as np
    import dreamer as D
    import tools
    import envs.minigrid as minigrid
    from nm512_margin_probe import MarginLogger

    overrides = dict(
        task=args.task_a, size=[64, 64], device=args.device, compile=False,
        video_pred_log=False, seed=args.seed,
        steps=int(1e9), eval_episode_num=args.eval_episodes,
        # DoorKey is sparse-reward: under greedy the reward head collapses to ~0 (the actor-critic
        # never gets a gradient) unless exploration finds the goal — a basin lottery. Plan2Explore
        # seeks novel key/door/goal states, populating replay with reward so the task actor can learn.
        # Eval always uses the task actor (training=False), so P2E only shapes data collection.
        # expl_extr_scale>0 = BALANCED P2E (explorer pursues task reward + curiosity, Continual-Dreamer
        # recipe). Pure curiosity (=0) wandered off-goal on DoorKey and starved the reward head.
        expl_behavior=args.expl_behavior, expl_until=args.expl_until,
        expl_extr_scale=args.expl_extr_scale,
    )
    if args.smoke:
        overrides.update(prefill=30, eval_episode_num=1, dataset_size=2000, batch_size=4,
                         batch_length=16, train_ratio=8)
    config = build_config(["minigrid"], overrides)
    assert config.action_repeat == 1, "orchestrator assumes action_repeat=1 (minigrid)"
    tools.set_seed_everywhere(config.seed)

    logdir = pathlib.Path(args.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    eval_every = 200 if args.smoke else int(args.eval_every or config.eval_every)
    a_max = 600 if args.smoke else args.a_max_steps
    b_max = 600 if args.smoke else args.b_max_steps
    K = 3  # evals in the competence / A_before / A_after windows
    comp_bar = args.competence_bar

    logger = _make_capture_logger(tools, logdir, 0)

    # ---- Phase A: fresh agent, train task A to stable competence -----------------------
    a_traindir = logdir / "A_train_eps"
    a_evaldir = logdir / "A_eval_eps"
    a_train_eps, a_eval_eps = OrderedDict(), OrderedDict()  # simulate's eval cleanup needs popitem(last=)
    a_train_envs, a_eval_envs = build_task_envs(D, config, args.task_a)
    acts = a_train_envs[0].action_space  # OneHotAction -> Box(shape=(n,)); mirror dreamer.main
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    obs_space, act_space = a_train_envs[0].observation_space, acts

    print(f"[A] prefill {config.prefill} on {args.task_a}")
    random_prefill(tools, config, a_train_envs, a_train_eps, a_traindir, logger, config.prefill)

    a_dataset = D.make_dataset(a_train_eps, config)
    agent = D.Dreamer(obs_space, act_space, config, logger, a_dataset).to(config.device)
    agent.requires_grad_(False)

    a_eval_hist, a_step, state = [], 0, None
    while a_step < a_max:
        score = eval_return(tools, agent, a_eval_envs, a_eval_eps, a_evaldir, logger, config.eval_episode_num)
        a_eval_hist.append((a_step, score))
        print(f"[A] step {a_step} eval_return {score}")
        sustained = (len(a_eval_hist) >= K and
                     all(s is not None and s >= comp_bar for _, s in a_eval_hist[-K:]))
        if sustained and not args.smoke:
            print(f"[A] competence: last {K} evals >= {comp_bar} -> switch")
            break
        state = tools.simulate(agent, a_train_envs, a_train_eps, a_traindir, logger,
                               limit=config.dataset_size, steps=eval_every, state=state)
        a_step += eval_every

    A_before = float(np.mean([s for _, s in a_eval_hist[-K:] if s is not None]))
    competent = bool(a_eval_hist and a_eval_hist[-1][1] is not None and a_eval_hist[-1][1] >= comp_bar)
    print(f"[A] A_before={A_before:.3f} competent={competent}")

    # ---- Snapshot the FIXED A-probe BEFORE clearing the buffer -------------------------
    margin_logger = MarginLogger(agent, a_dataset, device=config.device)
    snap = margin_logger()
    gate = "PASS" if (snap["gate_ok"] and abs(snap["frozen_margin"] - snap["live_margin"]) < 1e-3) else "FAIL"
    print(f"[snapshot] frozen {snap['frozen_margin']:+.3f} live {snap['live_margin']:+.3f} "
          f"(GATE {gate}) | flip {snap['frozen_flip']:.3f} | wm_on_A {snap['wm_on_A_recon']:.4f}")
    if gate == "FAIL":
        print("[snapshot] *** RE-ENCODE GATE FAILED: live != frozen at competence -> probe mis-wired, "
              "Q2 results from this run are NOT trustworthy. ***")
    torch.save({"agent_state_dict": agent.state_dict(), "A_before": A_before,
                "baseline": snap, "a_eval_hist": a_eval_hist},
               logdir / "A_competent.pt")

    # ---- Switch: set up B-phase replay per --replay_mode -------------------------------
    b_evaldir = logdir / "B_eval_eps"
    b_eval_eps = OrderedDict()
    b_train_envs, b_eval_envs = build_task_envs(D, config, args.task_b)
    if args.replay_mode == "shared":
        # NEVER-CLEAR: ONE shared buffer -- A stays, B accumulates into it -> natural DECAYING A-fraction
        # (the Continual-Dreamer-style protocol). B-phase writes into a_train_eps / a_traindir.
        b_train_eps, b_traindir = a_train_eps, a_traindir
        print(f"[switch] SHARED buffer (never-clear); prefill {config.prefill} B into shared | "
              f"start A-eps {len(a_train_eps)}")
        random_prefill(tools, config, b_train_envs, b_train_eps, b_traindir, logger, config.prefill)
        agent._dataset = D.make_dataset(a_train_eps, config)
        eff_frac, n_a = -1.0, -1  # natural/decaying mix
    else:  # "clear" (default): fresh B buffer + FIXED A-rehearsal fraction; A buffer kept for mixing
        b_traindir = logdir / "B_train_eps"
        b_train_eps = OrderedDict()
        print(f"[switch] CLEAR B buffer; prefill {config.prefill} on {args.task_b}")
        random_prefill(tools, config, b_train_envs, b_train_eps, b_traindir, logger, config.prefill)
        agent._dataset, eff_frac, n_a = make_mixed_dataset(
            D, tools, a_train_eps, b_train_eps, config, args.replay_fraction)
        print(f"[switch] replay_fraction requested {args.replay_fraction} -> effective {eff_frac:.4f} "
              f"({n_a}/{config.batch_size} A-seqs per batch) | A-buffer episodes {len(a_train_eps)}")
        assert args.replay_fraction <= 0 or len(a_train_eps) > 0, "replay_fraction>0 but A-buffer is empty!"
    # margin_logger.A_batch stays frozen (captured from a_dataset at construction, before the switch)

    # ---- Actor-protection: snapshot the A-competent actor as a frozen A-eval head ------
    # frozen_a_head: deep-copy the actor NOW (at A-competence), freeze it, and use it ONLY for the
    # A-eval swap during phase B. The live actor keeps training on B unchanged. This isolates actor
    # drift out of A-retention: A-eval = obs -> live(B-drifted) WM -> FROZEN A-actor -> action, so the
    # residual A-loss is pure representation overwrite. (joint = None -> unchanged baseline.)
    frozen_a_actor = None
    if args.actor_mode == "frozen_a_head":
        frozen_a_actor = copy.deepcopy(agent._task_behavior.actor)
        for p in frozen_a_actor.parameters():
            p.requires_grad_(False)
        frozen_a_actor.eval()
        # GATE (pre-reg 5.1): before ANY B-training, a frozen-head A-eval must reproduce A_before
        # (nothing has changed yet). If not, the swap is mis-wired -> run is INVALID.
        gate_ret = eval_return_with_actor(tools, agent, frozen_a_actor, a_eval_envs, a_eval_eps,
                                          a_evaldir, logger, config.eval_episode_num)
        ok = gate_ret is not None and abs(gate_ret - A_before) <= 0.15
        print(f"[frozen_a_head] swap sanity gate: frozen A-eval {gate_ret} vs A_before {A_before:.3f} "
              f"-> {'PASS' if ok else 'FAIL'}")
        if not ok:
            print("[frozen_a_head] *** SWAP SANITY GATE FAILED: frozen head != A_before pre-B -> "
                  "pointer-swap mis-wired, this run is NOT trustworthy. ***")

    # ---- WM-protection: freeze the representation (encoder + dynamics) at A-competence -
    # frozen_rep: EXCLUDE encoder+dynamics from the WM optimizer. Reward/cont heads, decoder, critic
    # and actor stay trainable -- the actor trains in imagination against wm.heads['reward'], so
    # freezing the reward head would leave B optimizing A's reward and the arm would be meaningless.
    # NOTE a requires_grad_(False) freeze LEAKS here: WM._train wraps itself in tools.RequiresGrad,
    # which re-enables requires_grad on ALL WM params every train step (caught by the fingerprint gate
    # on the first smoke run). The robust freeze is optimizer-level: rebuild model_opt without the
    # frozen params -- Adam only steps params it owns. Frozen grads are still computed and enter the
    # global clip norm (grad_clip=1000, far from binding); heads/decoder get fresh Adam state at the
    # switch. Retention of A under frozen_rep+frozen_a_head is STRUCTURAL (no trainable component in
    # the A-eval path); the experiment measures B's plasticity through the frozen rep (pre-reg P1).
    wm_fp_switch = None
    if args.wm_mode == "frozen_rep":
        frozen_prefixes = ("encoder.", "dynamics.")
        trainable = [p for n, p in agent._wm.named_parameters() if not n.startswith(frozen_prefixes)]
        n_frz = sum(p.numel() for n, p in agent._wm.named_parameters() if n.startswith(frozen_prefixes))
        assert trainable and n_frz > 0, "frozen_rep: param name prefixes did not match encoder/dynamics"
        agent._wm._model_opt = tools.Optimizer(
            "model", trainable, config.model_lr, config.opt_eps, config.grad_clip,
            config.weight_decay, opt=config.opt, use_amp=agent._wm._use_amp)
        wm_fp_switch = wm_rep_fingerprint(agent._wm)
        print(f"[frozen_rep] rebuilt model_opt excluding encoder+dynamics ({n_frz} params frozen) "
              f"| fingerprint {wm_fp_switch}")

    # ---- Phase B: train B, measure A every eval ---------------------------------------
    rows, b_step, state = [], 0, None
    ab_path = logdir / "ab_metrics.jsonl"
    with ab_path.open("w") as f:
        f.write("")  # truncate
    while b_step < b_max:
        b_eval = eval_return(tools, agent, b_eval_envs, b_eval_eps, b_evaldir, logger, config.eval_episode_num)
        # A-eval uses the frozen A-head under frozen_a_head; the live actor under joint (frozen_a_actor=None).
        a_ret = eval_return_with_actor(tools, agent, frozen_a_actor, a_eval_envs, a_eval_eps,
                                       a_evaldir, logger, config.eval_episode_num)
        m = margin_logger()
        row = {"b_step": b_step, "A_retention": a_ret, "B_eval": b_eval,
               "frozen_margin": m["frozen_margin"], "frozen_flip": m["frozen_flip"],
               "live_margin": m["live_margin"], "live_flip": m["live_flip"],
               "wm_on_A_recon": m["wm_on_A_recon"]}
        rows.append(row)
        with ab_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[B] step {b_step} | A_ret {a_ret} | B_eval {b_eval} | "
              f"frozen {m['frozen_margin']:+.3f} | live {m['live_margin']:+.3f} | "
              f"live_flip {m['live_flip']:.3f} | wm_on_A {m['wm_on_A_recon']:.4f}")
        state = tools.simulate(agent, b_train_envs, b_train_eps, b_traindir, logger,
                               limit=config.dataset_size, steps=eval_every, state=state)
        b_step += eval_every

    # frozen_rep structural gate: encoder+dynamics bytes must be identical at switch and end.
    wm_fp_end, wm_frozen_ok = None, None
    if args.wm_mode == "frozen_rep":
        wm_fp_end = wm_rep_fingerprint(agent._wm)
        wm_frozen_ok = (wm_fp_end == wm_fp_switch)
        print(f"[frozen_rep] fingerprint gate: switch {wm_fp_switch} end {wm_fp_end} -> "
              f"{'PASS' if wm_frozen_ok else 'FAIL -- FREEZE LEAKED, run NOT trustworthy'}")

    A_after = float(np.mean([r["A_retention"] for r in rows[-K:] if r["A_retention"] is not None]))
    A_peak = float(np.max([r["A_retention"] for r in rows if r["A_retention"] is not None])) if rows else None
    B_final = float(np.mean([r["B_eval"] for r in rows[-K:] if r["B_eval"] is not None]))
    B_peak = float(np.max([r["B_eval"] for r in rows if r["B_eval"] is not None])) if rows else None
    last = rows[-1] if rows else {}
    # Pre-registered TWO-BAR success (BOTH required): A recovers AND B still learns (stability-plasticity).
    A_RECOVER_BAR, B_COMPETENCE_BAR = 0.6, 0.6
    a_recovers = A_after >= A_RECOVER_BAR
    b_learns = (B_peak is not None) and (B_peak >= B_COMPETENCE_BAR)
    success_both_bars = bool(a_recovers and b_learns)

    # ---- Q1/Q2 read (per-seed; verdict + cross-seed band come from aggregating n>=3) --
    # Q2 DECIDER = frozen-vs-live margin gap, NOT the wm_on_A ratio (kept only as a secondary proxy;
    # see STAGE_AB_LOCATE_RESULT note -- its ratio is inflated by a near-zero memorization baseline).
    summary = {
        "seed": config.seed, "task_a": args.task_a, "task_b": args.task_b,
        "replay_mode": args.replay_mode, "actor_mode": args.actor_mode, "wm_mode": args.wm_mode,
        "wm_rep_fingerprint_switch": wm_fp_switch, "wm_rep_fingerprint_end": wm_fp_end,
        "wm_frozen_ok": wm_frozen_ok,
        "replay_fraction_requested": args.replay_fraction, "replay_fraction_effective": eff_frac,
        "A_before": A_before, "A_after": A_after, "A_peak": A_peak, "forgetting_abs": A_before - A_after,
        "retention_frac": (A_after / A_before) if A_before > 1e-9 else None,
        "B_final": B_final, "B_peak": B_peak, "A_competent": competent,
        # PRE-REGISTERED two-bar success (BOTH required): A recovers >=0.6 AND B still learns (B_peak>=0.6)
        "a_recovers": a_recovers, "b_learns": b_learns, "success_both_bars": success_both_bars,
        "gate_ok": bool(snap["gate_ok"]) and abs(snap["frozen_margin"] - snap["live_margin"]) < 1e-3,
        # FROZEN = actor on cached competent latents (actor drift on GOOD latents)
        "baseline_frozen_margin": snap["frozen_margin"], "final_frozen_margin": last.get("frozen_margin"),
        "baseline_frozen_flip": snap["frozen_flip"], "final_frozen_flip": last.get("frozen_flip"),
        # LIVE = actor on latents re-encoded through the post-B WM (the disambiguator)
        "baseline_live_margin": snap["live_margin"], "final_live_margin": last.get("live_margin"),
        "final_live_flip": last.get("live_flip"),
        # secondary proxy only
        "baseline_wm_on_A": snap["wm_on_A_recon"], "final_wm_on_A": last.get("wm_on_A_recon"),
        "n_b_evals": len(rows),
    }
    with (logdir / "ab_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== AB LOCATE SUMMARY (one seed) ===")
    print(json.dumps(summary, indent=2))
    print(f"REPLAY two-bar success (BOTH required): A_recovers(A_after {A_after:.3f}>=0.6)={a_recovers} "
          f"AND B_learns(B_peak {B_peak}>=0.6)={b_learns} -> success={success_both_bars}. "
          "Replay is the BASELINE bar, not the contribution. Q2 mechanism: did frozen recover (actor fixed) "
          "or only wm_on_A (representation)? -- diagnostic only; success = real-env retention of BOTH tasks.")
    for env in a_train_envs + a_eval_envs + b_train_envs + b_eval_envs:
        try:
            env.close()
        except Exception:
            pass
    return summary


def _parser():
    p = argparse.ArgumentParser(description="A->B locate run (v1.0) on NM512.")
    p.add_argument("--logdir", required=True)
    p.add_argument("--task_a", default="minigrid_DoorKey-9x9")
    p.add_argument("--task_b", default="minigrid_SimpleCrossingS9N1")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--a_max_steps", type=int, default=200000)
    p.add_argument("--b_max_steps", type=int, default=150000)
    p.add_argument("--competence_bar", type=float, default=0.6)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--n_states", type=int, default=512)
    p.add_argument("--eval_every", type=int, default=0)  # 0 = config default (1e4); set ~2000 for dense lead-timing
    p.add_argument("--replay_fraction", type=float, default=0.0)  # f of each B-batch that is task-A rehearsal (0 = no-replay baseline)
    p.add_argument("--replay_mode", default="clear", choices=["clear", "shared"])  # clear=fixed-fraction mix; shared=never-clear (decaying)
    p.add_argument("--actor_mode", default="joint", choices=["joint", "frozen_a_head"])  # joint=unchanged; frozen_a_head=freeze A-actor, use for A-eval only
    p.add_argument("--wm_mode", default="joint", choices=["joint", "frozen_rep"])  # frozen_rep=freeze encoder+dynamics at switch (heads stay trainable)
    p.add_argument("--expl_behavior", default="plan2explore")  # DoorKey sparse-reward needs P2E (greedy = basin lottery)
    p.add_argument("--expl_until", type=int, default=1000000)   # >> total A+B steps -> P2E acts throughout
    p.add_argument("--expl_extr_scale", type=float, default=1.0)  # balanced P2E (intr+extr); pure curiosity wandered off-goal
    p.add_argument("--smoke", action="store_true")
    return p


if __name__ == "__main__":
    run(_parser().parse_args())
