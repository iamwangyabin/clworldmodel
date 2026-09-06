"""Chain locate-run: N tasks sequentially on ONE agent + ONE never-clear shared buffer.

Generalizes orchestrator_ab_nm512.py (which stays untouched — banked A->B results depend on it)
to a task LIST, using the validated shared-buffer recipe (STAGE_AB_SHARED_RESULT_2026-07-03:
3/3 both-bars at 2 tasks). Design + pre-reg: CHAIN_TEST_DESIGN_2026-07-03.md.

Per phase i:
  - prefill task-i experience into the SHARED buffer (never cleared),
  - train until ALL-GOOD early stop: current task AND every previous task >= bar for K consecutive
    evals (encodes the transient lesson from shared s2 — stopping on current-task competence alone
    would freeze a see-saw mid-swing), or until --phase_max_steps,
  - snapshot a per-task MarginLogger probe (frozen latents from THIS phase's episodes) for
    rep-vs-actor attribution as tasks stack.

Every eval_every steps: real-env eval of the current task and ALL previous tasks (the bar).
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parent, _HERE.parent.parent):
    if (_cand / "dreamer.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
        break

from orchestrator_ab_nm512 import (  # noqa: E402  (path bootstrap must run first)
    _make_capture_logger, build_config, build_task_envs, eval_return, random_prefill)


# --------------------------------------------------------------------------------------
# COMPOSITE (pre-reg COMPOSITE_INTEGRATED_PREREG_2026-07-06): task-label-free eval via
# prototype router (album centroids, provisional@1/commit@10) + margin-arbitrated policy
# selection between frozen per-task heads and the live actor. All behind --composite.
# --------------------------------------------------------------------------------------
def _episode_mean_embed(agent, ep):
    """Per-episode mean frame-embedding under the CURRENT encoder (albums store frames, not
    embeddings — re-embedded every round so router state never goes stale)."""
    import numpy as np
    import torch
    wm = agent._wm
    batch = {k: np.asarray(v)[None] for k, v in ep.items() if "log_" not in k}
    with torch.no_grad():
        data = wm.preprocess(batch)
        e = wm.encoder(data)  # [1, T, E]
    return e[0].mean(0).detach().cpu().numpy()


def _build_centroids(agent, albums):
    import numpy as np
    tids = sorted(albums)
    cents = np.stack([np.stack([_episode_mean_embed(agent, ep) for ep in albums[j]]).mean(0)
                      for j in tids])
    return cents, tids


def _build_policy_picks(agent, heads, albums, n_arb=8):
    """Label-free arbitration (post-hoc best-of finding): for each completed task j, pick
    {head_j, live} by mean log-likelihood of the album's recorded competent actions under
    latents re-encoded through the CURRENT WM. No env rollouts."""
    import numpy as np
    import torch
    from nm512_margin_probe import _encode_feats
    live = agent._task_behavior.actor
    picks, pick_names = {}, {}
    for j, head in heads.items():
        lps = {"head": [], "live": []}
        for ep in albums[j][:n_arb]:
            batch = {k: np.asarray(v)[None] for k, v in ep.items() if "log_" not in k}
            feat = _encode_feats(agent, batch)
            actions = torch.as_tensor(np.asarray(ep["action"])[None], dtype=feat.dtype,
                                      device=feat.device)
            valid = actions.sum(-1) > 0.5  # reset steps store all-zero actions -> mask them
            if not bool(valid.any()):
                continue
            with torch.no_grad():
                for name, mod in (("head", head), ("live", live)):
                    logp = torch.log(mod(feat).probs + 1e-9)  # avoid one-hot validation path
                    lp = (logp * actions).sum(-1)[valid].mean()
                    lps[name].append(float(lp))
        use_head = bool(lps["head"]) and float(np.mean(lps["head"])) >= float(np.mean(lps["live"]))
        picks[j] = head if use_head else live
        pick_names[j] = "head" if use_head else "live"
    return picks, pick_names


def real_bc_update(agent, batch):
    """Real-data behavior-cloning rehearsal (pre-reg REAL_BC_ABLATION_2026-07-16): clone the
    RECORDED actions of stored prior-task episodes — no imagination, no grading. This is the
    CLEAR-style alternative, matched update-for-update against tunnel_update. Actor BC only."""
    import torch

    import tools as tools_mod
    wm, behavior = agent._wm, agent._task_behavior
    with torch.no_grad():
        data = wm.preprocess(batch)
        embed = wm.encoder(data)
        post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
        feat = wm.dynamics.get_feat(post).detach()          # [B, T, F]
        act = data["action"].detach()                        # [B, T, A]
        # BUGFIX 2026-07-18 — TEMPORAL ALIGNMENT (THE bug: real_bc destroyed a skill while cloning
        # its OWN success, DoorKey 0.965->0.0). The RSSM's observe() action convention lags the
        # actor's decision by one step: the actor's argmax at feat[t] matches the RECORDED action
        # at t+1 (measured agreement 0.96 shifted vs 0.30 unshifted). So BC feat[t] toward act[t+1].
        # Unshifted, the actor was cloned toward actions it never took (0.29 prob) -> collapse.
        # (tunnel_update never hit this: imagined feat+action are generated together, aligned.)
        feat = feat[:, :-1]
        act = act[:, 1:]
        valid = act.sum(-1) > 0.5                            # non-reset steps (all-zero actions masked)
    with tools_mod.RequiresGrad(behavior.actor):
        dist = behavior.actor(feat)
        act_safe = act.clone()
        reset = ~valid                                # [B, T-1]
        act_safe[reset] = 0.0
        act_safe[reset, 0] = 1.0                      # dummy action-0 one-hot at reset steps (masked)
        lp = dist.log_prob(act_safe)                  # [B, T-1]
        loss = -(lp * valid.float()).sum() / valid.float().sum().clamp(min=1.0)
        behavior._actor_opt(loss, behavior.actor.parameters())
    return float(loss.detach())


def tunnel_update(agent, batch, topk_frac=0.25, cont_grading=False):
    """One dream self-imitation step (pre-reg TUNNEL_CHAIN_PREREG_2026-07-12; mechanism banked
    3/3 in DREAM_RECOVERY_RESULT_2026-07-12): imagine from replayed old-task starts with the
    sampling actor, grade each trajectory with the live reward head + critic bootstrap,
    behavior-clone the actor on the top fraction. Actor BC step ONLY — critic and WM train
    exactly as in the plain chain."""
    import torch

    import tools as tools_mod
    wm, behavior = agent._wm, agent._task_behavior
    cfg = behavior._config
    with torch.no_grad():
        data = wm.preprocess(batch)
        embed = wm.encoder(data)
        post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
        start = {k: v.detach() for k, v in post.items()}
        feats, _states, actions = behavior._imagine(start, behavior.actor, cfg.imag_horizon)
        r = wm.heads["reward"](feats).mode().squeeze(-1)
        gam = cfg.discount ** torch.arange(r.shape[0], device=r.device,
                                           dtype=r.dtype).unsqueeze(-1)
        v_last = behavior.value(feats[-1]).mode().squeeze(-1)
        if cont_grading:
            # Fix A v2.2 "lex" (pre-reg 2026-07-13, amended 2026-07-14 x2): realized-first
            # dream grading. (1) No scoring past the dream's own termination: rewards weighted
            # by REACH probability, bootstrap by survival — kills the post-terminal noise that
            # made T2 selection near-random. (2) Dreams that ACTUALLY achieved reward outrank
            # value promises — kills the promise-beats-late-success failure that stalled pure
            # shift on recovery (referee: shift flat 0.0 @8.5k; lex PASS @2k, ~2x faster than
            # the banked dep scorer). Offline gauge: AUC 1.0 / purity 1.0 on both task types.
            cont = wm.heads["cont"](feats).mean.squeeze(-1)
            surv = torch.cumprod(cont, dim=0)
            reach = torch.cat([torch.ones_like(cont[:1]), surv[:-1]], dim=0)
            realized = (gam * reach * r).sum(0)
            score = (realized > 0.3).float() * 10.0 + realized \
                + cfg.discount ** r.shape[0] * surv[-1] * v_last
        else:
            score = (gam * r).sum(0) + cfg.discount ** r.shape[0] * v_last
        k = max(1, int(topk_frac * score.shape[0]))
        top = score.topk(k).indices
        bc_feat, bc_act = feats[:, top].detach(), actions[:, top].detach()
    with tools_mod.RequiresGrad(behavior.actor):
        dist = behavior.actor(bc_feat)
        loss = -(dist.log_prob(bc_act)).mean()
        behavior._actor_opt(loss, behavior.actor.parameters())
    return float(loss.detach())


class RoutedEvalPolicy:
    """Eval policy with NO task label: routes each episode by nearest album centroid on the
    running-mean frame embedding (provisional from step 1, committed at commit_k — probe v4),
    then acts with the arbitrated policy for the routed task. try/finally actor swap."""

    def __init__(self, agent, centroids, task_ids, policy_by_task, commit_k=10):
        self.agent, self.centroids, self.task_ids = agent, centroids, task_ids
        self.policy_by_task, self.commit_k = policy_by_task, commit_k
        self._sum, self._n, self._pick = None, 0, None
        self.episode_picks = []

    def _embed(self, obs):
        import torch
        wm = self.agent._wm
        with torch.no_grad():
            data = wm.preprocess(obs)
            e = wm.encoder(data)  # [B, E]
        return e[0].detach().cpu().numpy()

    def __call__(self, obs, reset, state=None, training=False):
        import numpy as np
        if bool(np.asarray(obs["is_first"]).flatten()[0]):
            self.flush()
            self._sum, self._n = None, 0
        emb = self._embed(obs)
        self._sum = emb if self._sum is None else self._sum + emb
        self._n += 1
        if self._pick is None or self._n <= self.commit_k:  # provisional until commit
            d = np.linalg.norm(self.centroids - (self._sum / self._n)[None], axis=1)
            self._pick = self.task_ids[int(d.argmin())]
        # SAFE OPENING (comp4_s1 lesson): act with the LIVE actor until the route commits, then
        # switch ONCE. Mid-episode head-flapping during the provisional window killed the lethal
        # task (LavaGap 0.94 -> 0.28) while router accuracy was 97.7% — the switching was the
        # damage, not the routing.
        if self._n < self.commit_k:
            return self.agent(obs, reset, state, training=False)
        tb = self.agent._task_behavior
        live = tb.actor
        try:
            tb.actor = self.policy_by_task[self._pick]
            out, st = self.agent(obs, reset, state, training=False)
        finally:
            tb.actor = live
        return out, st

    def flush(self):
        if self._pick is not None:
            self.episode_picks.append(self._pick)
            self._pick = None


def eval_return_routed(tools, routed_policy, eval_envs, eval_eps, evaldir, logger, episodes):
    logger.captured.pop("eval_return", None)
    tools.simulate(routed_policy, eval_envs, eval_eps, evaldir, logger,
                   is_eval=True, episodes=episodes)
    routed_policy.flush()
    return logger.captured.get("eval_return")


def run(args):
    from collections import OrderedDict
    import numpy as np
    import torch
    import dreamer as D
    import tools
    from nm512_margin_probe import MarginLogger

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    assert len(tasks) >= 1, "chain needs >= 1 task (1 = solo audition run)"

    overrides = dict(
        task=tasks[0], size=[64, 64], device=args.device, compile=False,
        video_pred_log=False, seed=args.seed,
        steps=int(1e9), eval_episode_num=args.eval_episodes,
        expl_behavior=args.expl_behavior, expl_until=0, expl_extr_scale=1.0,
    )
    if args.smoke:
        overrides.update(prefill=30, eval_episode_num=1, dataset_size=2000, batch_size=4,
                         batch_length=16, train_ratio=8)
    config = build_config(["minigrid"], overrides)
    assert config.action_repeat == 1, "chain orchestrator assumes action_repeat=1 (minigrid)"
    tools.set_seed_everywhere(config.seed)

    logdir = pathlib.Path(args.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    eval_every = 200 if args.smoke else args.eval_every
    phase_max = 600 if args.smoke else args.phase_max_steps
    K, bar = 3, args.competence_bar

    logger = _make_capture_logger(tools, logdir, 0)

    # ---- ONE shared buffer for the whole chain (the validated never-clear mode) ---------
    shared_eps = OrderedDict()
    traindir = logdir / "train_eps"

    # Per-task handles, created lazily at each phase start and kept open for later evals.
    task_train_envs, task_eval_envs, task_eval_eps, task_evaldirs = [], [], [], []

    agent, state = None, None
    probes = []            # MarginLogger per completed phase (frozen probe on that task's episodes)
    phase_records = []     # per-phase summary rows
    rows = []              # every eval row across the whole chain
    heads, albums = {}, {}         # composite: frozen per-task heads + prototype albums
    rehearsal_data = {}            # tunnel_rehearsal: j -> dataset over task-j's own-phase eps
    rehearsal_losses = []          # rolling BC losses, logged per chunk
    router_hits = router_total = 0  # in-phase router accuracy vs ground truth (pooled)
    # 2026-07-10 audit: pooled accuracy hid a 0.40 per-task collapse behind an 0.75 overall.
    # Track per-task from now on; pooled kept for continuity with banked runs.
    router_by = {}  # j -> [hits, total], in-phase
    # Real-env bandit arbitration (comp4_s1: album-statistic metrics are razor-thin noise while the
    # real-env gap is 2x — held-state stats can't predict closed-loop competence; returns can).
    bandit = {}  # j -> {"head": [returns], "live": [returns]}

    def bandit_arm(j):
        import numpy as np
        b = bandit.setdefault(j, {"head": [], "live": []})
        if len(b["head"]) < 3 or len(b["live"]) < 3:  # explore both arms first
            return "head" if len(b["head"]) <= len(b["live"]) else "live"
        return "head" if float(np.mean(b["head"])) >= float(np.mean(b["live"])) else "live"
    metrics_path = logdir / "chain_metrics.jsonl"
    metrics_path.write_text("")

    for i, task in enumerate(tasks):
        train_envs, eval_envs = build_task_envs(D, config, task)
        task_train_envs.append(train_envs)
        task_eval_envs.append(eval_envs)
        task_eval_eps.append(OrderedDict())
        evaldir = logdir / f"eval_eps_{i}_{task.replace('/', '_')}"
        task_evaldirs.append(evaldir)

        acts = train_envs[0].action_space
        n_act = acts.n if hasattr(acts, "n") else acts.shape[0]
        if i == 0:
            config.num_actions = n_act
            obs_space, act_space = train_envs[0].observation_space, acts
            print(f"[phase 0] prefill {config.prefill} on {task}")
            random_prefill(tools, config, train_envs, shared_eps, traindir, logger, config.prefill)
            agent = D.Dreamer(obs_space, act_space, config, logger,
                              D.make_dataset(shared_eps, config)).to(config.device)
            agent.requires_grad_(False)
        else:
            assert n_act == config.num_actions, f"action-space mismatch at {task}"
            print(f"[phase {i}] prefill {config.prefill} on {task} into SHARED buffer "
                  f"({len(shared_eps)} episodes so far)")
            random_prefill(tools, config, train_envs, shared_eps, traindir, logger, config.prefill)
            agent._dataset = D.make_dataset(shared_eps, config)

        eps_before_phase = set(shared_eps.keys())
        phase_rows, phase_step, state = [], 0, None  # fresh sim state per phase (new envs)
        # Re-graduation (pre-reg 2026-07-14, Fix C rung): require N non-overlapping all-good
        # windows before advancing; harvest the rehearsal library from those windows only.
        grad_windows, rows_since_grad, keys_at_row = [], 0, []
        while phase_step < phase_max:
            if args.composite and heads:  # router state rebuilt each round (drift-proof)
                cents, tids = _build_centroids(agent, albums)
                round_arms = {k: bandit_arm(k) for k in heads}
                policy_by_task = {k: (heads[k] if round_arms[k] == "head"
                                      else agent._task_behavior.actor) for k in heads}
            rets = []
            for j in range(i + 1):
                if args.composite and j in heads:
                    rp = RoutedEvalPolicy(agent, cents, tids, policy_by_task, commit_k=10)
                    r = eval_return_routed(tools, rp, task_eval_envs[j], task_eval_eps[j],
                                           task_evaldirs[j], logger, config.eval_episode_num)
                    router_hits += sum(1 for p in rp.episode_picks if p == j)
                    router_total += len(rp.episode_picks)
                    rb = router_by.setdefault(j, [0, 0])
                    rb[0] += sum(1 for p in rp.episode_picks if p == j)
                    rb[1] += len(rp.episode_picks)
                    if r is not None:  # audit fix: None used to enter arm means as 0.0
                        bandit[j][round_arms[j]].append(r)
                else:
                    r = eval_return(tools, agent, task_eval_envs[j], task_eval_eps[j],
                                    task_evaldirs[j], logger, config.eval_episode_num)
                rets.append(r)
            row = {"phase": i, "task": task, "phase_step": phase_step,
                   "chain_step": sum(p["steps_used"] for p in phase_records) + phase_step}
            for j, r in enumerate(rets):
                row[f"ret_{j}"] = r
            for j, probe in enumerate(probes):
                m = probe()
                row[f"m{j}_frozen"] = m["frozen_margin"]
                row[f"m{j}_live"] = m["live_margin"]
            rows.append(row)
            phase_rows.append(row)
            with metrics_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[P{i} {task}] step {phase_step} | " +
                  " | ".join(f"T{j} {r if r is None else round(r, 2)}" for j, r in enumerate(rets)))

            # ALL-GOOD early stop: last K evals, every task 0..i at/above bar.
            keys_at_row.append(set(shared_eps.keys()))
            rows_since_grad += 1
            window = phase_rows[-K:]
            all_good = (len(window) == K and all(
                w.get(f"ret_{j}") is not None and w[f"ret_{j}"] >= bar
                for w in window for j in range(i + 1)))
            if all_good and rows_since_grad >= K:
                base = keys_at_row[-K - 1] if len(keys_at_row) > K else eps_before_phase
                grad_windows.append({"end_step": phase_step,
                                     "keys": set(shared_eps.keys()) - base})
                rows_since_grad = 0
                print(f"[P{i}] graduation {len(grad_windows)}/{args.graduations} at step {phase_step}")
            if len(grad_windows) >= args.graduations and all_good and not args.smoke:
                print(f"[P{i}] ALL-GOOD x{len(grad_windows)}: tasks 0..{i} >= {bar} -> next phase")
                break
            state = tools.simulate(agent, train_envs, shared_eps, traindir, logger,
                                   limit=config.dataset_size, steps=eval_every, state=state)
            phase_step += eval_every
            if args.tunnel_rehearsal and rehearsal_data:
                chunk_losses = []
                for j, ds in rehearsal_data.items():
                    for _ in range(args.rehearsal_updates):
                        if args.rehearsal_source == "dream":
                            chunk_losses.append(tunnel_update(
                                agent, next(ds), cont_grading=args.cont_grading))
                        else:  # real / real_filtered: clone recorded actions (ablation arm)
                            chunk_losses.append(real_bc_update(agent, next(ds)))
                rehearsal_losses.append(float(np.mean(chunk_losses)))
                print(f"[P{i} rehearsal] {len(chunk_losses)} tunnel updates over tasks "
                      f"{sorted(rehearsal_data)} | mean BC loss {rehearsal_losses[-1]:.4f}")

        phase_records.append({"phase": i, "task": task, "steps_used": phase_step,
                              "all_good": bool(all_good),
                              "graduations": len(grad_windows)})

        if args.composite:  # snapshot the head + prototype album at task-i competence
            head = copy.deepcopy(agent._task_behavior.actor)
            for p in head.parameters():
                p.requires_grad_(False)
            head.eval()
            heads[i] = head
            # simulate prunes the in-memory eval cache to ~1 episode; the full history is on DISK
            disk_eps = tools.load_episodes(task_evaldirs[i], limit=10 ** 9)
            album_src = [e for e in disk_eps.values() if len(next(iter(e.values()))) >= 3]
            # load_episodes returns NEWEST-FIRST, so [:20] = the 20 most recent (competent)
            # episodes. [-20:] silently took the OLDEST 20 (random-policy rollouts from phase
            # start) — found 2026-07-10; routing_confusion_posthoc.py reproduces comp4b_s2's
            # router_acc_final=0.75 exactly under that rule (T0->T3 60%, T2->T0 38%).
            albums[i] = album_src[:20]
            print(f"[composite] snapshot head_{i} + album ({len(albums[i])} eps)")

        # Frozen probe for THIS task, from episodes collected during THIS phase.
        phase_eps = OrderedDict((k, v) for k, v in shared_eps.items() if k not in eps_before_phase)
        if len(phase_eps) >= 2:
            probes.append(MarginLogger(agent, D.make_dataset(phase_eps, config),
                                       device=config.device))
        if args.tunnel_rehearsal and args.rehearsal_source == "real_filtered":
            # strongest fair real-data alternative: competent episodes only (pre-reg
            # REAL_BC_ABLATION_2026-07-16) — removes the "you cloned beginner flailing" objection
            comp = OrderedDict((k, v) for k, v in phase_eps.items()
                               if float(np.asarray(v["reward"]).sum()) > 0.05)
            if len(comp) >= 2:
                rehearsal_data[i] = D.make_dataset(comp, config)
                print(f"[real_filtered] rehearsal dataset for task {i}: {len(comp)} competent "
                      f"of {len(phase_eps)} phase eps")
            else:
                rehearsal_data[i] = D.make_dataset(OrderedDict(phase_eps), config)
                print(f"[real_filtered] task {i}: only {len(comp)} competent eps — "
                      f"FALLBACK to all {len(phase_eps)} (logged)")
        elif args.tunnel_rehearsal and len(phase_eps) >= 2:
            if args.graduations > 1 and grad_windows:
                # Fix C harvest: library = episodes from the graduation windows ONLY
                wkeys = set().union(*(g["keys"] for g in grad_windows))
                lib = OrderedDict((k, v) for k, v in phase_eps.items() if k in wkeys)
                if len(lib) < 2:
                    lib = OrderedDict(phase_eps)  # fallback, logged
                rehearsal_data[i] = D.make_dataset(lib, config)
                print(f"[tunnel] rehearsal dataset for task {i}: {len(lib)} eps "
                      f"from {len(grad_windows)} graduation windows (of {len(phase_eps)} phase eps)")
            elif args.rehearsal_anchor > 0:
                # storage ablation: keep only the K
                # most recent COMPETENT episodes as the frozen rehearsal anchor, not the full
                # (growing) phase buffer. phase_eps is insertion-ordered oldest->newest.
                comp = [(k, v) for k, v in phase_eps.items()
                        if float(np.asarray(v["reward"]).sum()) > 0.05]
                anchor = OrderedDict(comp[-args.rehearsal_anchor:]) if comp \
                    else OrderedDict(list(phase_eps.items())[-args.rehearsal_anchor:])
                rehearsal_data[i] = D.make_dataset(anchor, config)
                print(f"[tunnel] rehearsal ANCHOR for task {i}: {len(anchor)} eps "
                      f"(K={args.rehearsal_anchor}, of {len(phase_eps)} phase eps, "
                      f"{len(comp)} competent)")
            else:
                # snapshot dict: rehearsal batches sample ONLY this task's own-phase episodes
                rehearsal_data[i] = D.make_dataset(OrderedDict(phase_eps), config)
                print(f"[tunnel] rehearsal dataset for task {i}: {len(phase_eps)} eps")
        torch.save({"agent_state_dict": agent.state_dict(), "phase": i, "task": task},
                   logdir / f"chain_phase{i}.pt")

    # ---- Final read: last-K rows of the final phase, ALL tasks ------------------------
    final_rows = rows[-K:]
    final_ret = {}
    for j, task in enumerate(tasks):
        vals = [r.get(f"ret_{j}") for r in final_rows if r.get(f"ret_{j}") is not None]
        final_ret[task] = float(np.mean(vals)) if vals else None
    success_all = all(v is not None and v >= bar for v in final_ret.values())

    # Per-task solve-stability over every eval AFTER that task's own phase.
    stability = {}
    for j, task in enumerate(tasks):
        later = [r for r in rows if r["phase"] > j and r.get(f"ret_{j}") is not None]
        stability[task] = (sum(1 for r in later if r[f"ret_{j}"] >= bar), len(later))

    # ---- COMPOSITE final read: K extra fully-routed rounds of ALL tasks ----------------
    final_routed, routed_success, router_acc_final, arb = None, None, None, None
    if args.composite:
        cents, tids = _build_centroids(agent, albums)
        fr = {j: [] for j in range(len(tasks))}
        fh = ft = 0
        fh_by = {j: [0, 0] for j in range(len(tasks))}  # audit: per-task final accuracy
        for _ in range(K + 3):  # extra rounds so the last task's cold bandit gets explore+exploit
            round_arms = {k: bandit_arm(k) for k in heads}
            policy_by_task = {k: (heads[k] if round_arms[k] == "head"
                                  else agent._task_behavior.actor) for k in heads}
            for j in range(len(tasks)):
                rp = RoutedEvalPolicy(agent, cents, tids, policy_by_task, commit_k=10)
                r = eval_return_routed(tools, rp, task_eval_envs[j], task_eval_eps[j],
                                       task_evaldirs[j], logger, config.eval_episode_num)
                fr[j].append(r)
                fh += sum(1 for p in rp.episode_picks if p == j)
                ft += len(rp.episode_picks)
                fh_by[j][0] += sum(1 for p in rp.episode_picks if p == j)
                fh_by[j][1] += len(rp.episode_picks)
                if r is not None:  # audit fix: None used to enter arm means as 0.0
                    bandit[j][round_arms[j]].append(r)
        arb = {str(j): {a: round(float(np.mean(v)), 3) if v else None
                        for a, v in b.items()} for j, b in bandit.items()}
        fr_all = {tasks[j]: v for j, v in fr.items()}  # every round incl. exploration (spread)
        fr = {j: v[-K:] for j, v in fr.items()}  # read the LAST K rounds (post-exploration)
        final_routed = {tasks[j]: (float(np.mean([x for x in v if x is not None]))
                                   if any(x is not None for x in v) else None)
                        for j, v in fr.items()}
        routed_success = all(v is not None and v >= bar for v in final_routed.values())
        router_acc_final = fh / max(1, ft)
        print(f"[composite] FINAL ROUTED: {final_routed} | router acc {router_acc_final:.3f} "
              f"| arbitration {arb}")

    summary = {
        "tasks": tasks, "seed": config.seed, "bar": bar,
        "phases": phase_records,
        "final_retention": final_ret,
        "success_all_tasks": success_all,
        "solve_stability": {t: f"{a}/{b}" for t, (a, b) in stability.items()},
        "n_evals": len(rows),
        "composite": bool(args.composite),
        "final_retention_routed": final_routed,
        "success_all_tasks_routed": routed_success,
        "router_accuracy_final": router_acc_final,
        "router_accuracy_inphase": (router_hits / router_total) if router_total else None,
        "router_accuracy_final_by_task": ({tasks[j]: (h / t if t else None)
                                           for j, (h, t) in fh_by.items()}
                                          if args.composite else None),
        "router_accuracy_inphase_by_task": {tasks[j]: (h / t if t else None)
                                            for j, (h, t) in router_by.items()},
        "final_routed_rounds": fr_all if args.composite else None,  # per-round spread, incl. exploration
        "arbitration_bandit": arb,
        "tunnel_rehearsal": bool(args.tunnel_rehearsal),
        "rehearsal_source": args.rehearsal_source if args.tunnel_rehearsal else None,
        "rehearsal_anchor": args.rehearsal_anchor if args.tunnel_rehearsal else None,
        "rehearsal_updates": args.rehearsal_updates if args.tunnel_rehearsal else None,
        "rehearsal_bc_loss_curve": rehearsal_losses if args.tunnel_rehearsal else None,
    }
    with (logdir / "chain_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== CHAIN SUMMARY (one seed) ===")
    print(json.dumps(summary, indent=2))
    # Audit 2026-07-10: the verdict line used to print ONLY the oracle metric; under --composite
    # the registered primary is the ROUTED read — print both, routed first.
    if args.composite:
        print(f"CHAIN routed success (PRIMARY, ALL {len(tasks)} tasks >= {bar}): {routed_success} "
              f"| oracle success (upper bound): {success_all}. Per-task solve-stability + "
              "transient lengths are the read, not just the endpoint.")
    else:
        print(f"CHAIN success (ALL {len(tasks)} tasks >= {bar} at end): {success_all}. "
              "Per-task solve-stability + transient lengths are the read, not just the endpoint.")

    for envs in task_train_envs + task_eval_envs:
        for env in envs:
            try:
                env.close()
            except Exception:
                pass
    return summary


def _parser():
    p = argparse.ArgumentParser(description="Chain locate-run (A->B->C->...) on NM512, shared buffer.")
    p.add_argument("--logdir", required=True)
    p.add_argument("--tasks", required=True,
                   help="comma-separated, e.g. minigrid_DoorKey-5x5,minigrid_SimpleCrossingS9N1,...")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--phase_max_steps", type=int, default=100000)
    p.add_argument("--competence_bar", type=float, default=0.6)
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--expl_behavior", default="greedy")
    p.add_argument("--composite", action="store_true")  # router + heads + arbitration (pre-reg 2026-07-06)
    p.add_argument("--tunnel_rehearsal", action="store_true")  # pre-reg 2026-07-12: dream self-imitation of prior tasks
    p.add_argument("--rehearsal_updates", type=int, default=50)  # tunnel updates per prior task per chunk
    p.add_argument("--cont_grading", action="store_true")  # pre-reg 2026-07-13 Fix A: survival-weighted dream scores
    p.add_argument("--graduations", type=int, default=1)  # Fix C rung: N non-overlapping all-good windows per phase
    p.add_argument("--rehearsal_source", choices=["dream", "real", "real_filtered"],
                   default="dream")  # pre-reg REAL_BC_ABLATION_2026-07-16: what the actor imitates
    p.add_argument("--rehearsal_anchor", type=int, default=0)  # storage ablation: keep only the K most recent competent episodes (0 = full buffer)
    p.add_argument("--smoke", action="store_true")
    return p


if __name__ == "__main__":
    run(_parser().parse_args())
