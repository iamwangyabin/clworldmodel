"""Stage A->B Q2 instrumentation, ported to the NM512/dreamerv3-torch trainer.

PURPOSE (the live A->B logger, mirrors cradle/diagnostics/ab_margin_probe.py semantics):
  at a FIXED held set of A-critical-states S with the A-competent policy's actions a*(s),
  and a FIXED held A-batch, each eval-interval during B-training measure
    margin(s)     = logp_pi(a*(s)) - max_{a != a*} logp_pi(a)    (>0 a* preferred, <0 flipped)
    flip_fraction = mean_s[ argmax pi(.|s) != a*(s) ]
    wm_on_A_recon = current WM recon loss (-decoder.log_prob) on the held A-batch
  -> margin collapses + WM-on-A flat  = actor-drift  (behavior dies, representation survives)
  -> WM-on-A grows                    = representation overwrite

WHY NO SINGLE-TASK DISCRIMINATION GATE HERE: the old script validated the logger by
reproducing a KNOWN good-vs-collapsed flip on a single task. On NM512 the single-task run
is STABLE (no collapsed ckpt exists), so that gate is N/A. Per the locked plan, the logger's
real validation IS the A->B run: A's margin must drop iff A's eval-return drops. This module
instead ships a __main__ SELF-TEST: load the competent ckpt and assert the logger is
self-consistent (margin>0 and flip~0 at a*, wm finite) -- catches wiring bugs before the run.

ACCESS POINTS (verified against NM512 commit in dreamerv3-torch-ref):
  actor : agent._task_behavior.actor(feat) -> tools.OneHotDist (.probs / .logits)
  WM    : agent._wm.preprocess / .encoder / .dynamics.obs_step / .get_feat / .heads['decoder']
  ckpt  : torch.load(logdir/'latest.pt')['agent_state_dict']  (keys carry '_orig_mod.' if
          trained with config.compile=True -> stripped here so we can load uncompiled)

RUN (from inside ~/projects/dreamerv3-torch on Spark):
  python nm512_margin_probe.py --self_test --logdir ~/dv3_logs/mg_proof_s1 --device cuda
"""
from __future__ import annotations

import argparse
import pathlib
import sys

# This module lives in cradle/diagnostics/ on the laptop but is scp'd next to dreamer.py on
# Spark. Make the trainer importable whether run from its dir or alongside it.
_HERE = pathlib.Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parent, _HERE.parent.parent):
    if (_cand / "dreamer.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
        break


# --------------------------------------------------------------------------------------
# Config + agent loading (mirrors dreamer.__main__ lines 347-370 and dreamer.main 292-304)
# --------------------------------------------------------------------------------------
def _recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            _recursive_update(base[key], value)
        else:
            base[key] = value


def build_config(config_names, logdir, device, extra_argv=None):
    """Reconstruct the same argparse Namespace dreamer.py builds, for the given config stack."""
    import ruamel.yaml as yaml
    import tools

    configs = yaml.safe_load((pathlib.Path(sys.modules["tools"].__file__).parent
                              / "configs.yaml").read_text())
    name_list = ["defaults", *config_names]
    defaults = {}
    for name in name_list:
        _recursive_update(defaults, configs[name])

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    config = parser.parse_args(extra_argv or [])

    config.logdir = str(logdir)
    config.device = device
    # mirror dreamer.main's path defaults so we can load the run's own episodes/ckpt
    logp = pathlib.Path(logdir).expanduser()
    config.traindir = config.traindir or str(logp / "train_eps")
    config.evaldir = config.evaldir or str(logp / "eval_eps")
    return config


def _strip_compile_prefix(state_dict):
    """Trained with config.compile=True (Linux) -> keys contain '_orig_mod.'; remove it so the
    same weights load into an uncompiled agent. No-op if trained uncompiled (Windows)."""
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


def load_agent(config, device):
    """Build a Dreamer agent and load latest.pt. Returns (agent, base_env_maker, dataset).

    base_env_maker(seed) yields the *unwrapped* MiniGrid env (Discrete actions, dict obs) for
    rolling the policy by hand in capture_A_states -- avoids depending on the wrapper-stack's
    step/reset signature.
    """
    import functools
    import torch
    import dreamer as D
    import tools
    import envs.minigrid as minigrid

    # compile breaks attribute access patterns we rely on AND is slow for an inference probe.
    config.compile = False

    # spaces + num_actions exactly as dreamer.main derives them (from the wrapped env)
    wrapped = D.make_env(config, "eval", 0)
    obs_space, act_space = wrapped.observation_space, wrapped.action_space
    config.num_actions = act_space.n if hasattr(act_space, "n") else act_space.shape[0]
    try:
        wrapped.close()
    except Exception:
        pass

    logger = tools.Logger(pathlib.Path(config.logdir).expanduser(), 0)
    train_eps = tools.load_episodes(pathlib.Path(config.traindir).expanduser(),
                                    limit=config.dataset_size)
    dataset = D.make_dataset(train_eps, config)

    agent = D.Dreamer(obs_space, act_space, config, logger, dataset).to(device)
    agent.requires_grad_(False)
    ckpt_path = pathlib.Path(config.logdir).expanduser() / "latest.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    agent.load_state_dict(_strip_compile_prefix(ckpt["agent_state_dict"]))
    agent.eval()

    suite, task = config.task.split("_", 1)
    assert suite == "minigrid", f"this probe is MiniGrid-only, got {suite}"
    env_maker = functools.partial(minigrid.MiniGrid, task, config.size)
    return agent, env_maker, dataset


# --------------------------------------------------------------------------------------
# Core measurements (semantics identical to ab_margin_probe.py, re-wired to NM512)
# --------------------------------------------------------------------------------------
def _encode_feats(agent, batch):
    """Preprocess + encode + DETERMINISTIC recurrent-observe through the CURRENT WM.
    The RSSM posterior is stochastic and RSSM.observe() samples through the WHOLE recurrence
    (each obs_step defaults sample=True), so a sampled encode is NOT reproducible -- two encodes
    of the same batch through the same WM differ, which would make frozen-vs-live measure SAMPLE
    NOISE instead of WM change (this is what tripped the validation gate). We instead run the
    posterior MODE at every step (obs_step sample=False), mirroring observe's scan exactly. Then:
    identical WM -> identical feats (the gate invariant), and feats move ONLY when the WM changes.
    Latents are recurrent (a single obs does not determine its latent), so the full sequence is run.
    Always via wm.preprocess so the encode path is byte-identical to training (no preprocess drift).
    Returns feat [B, T, F]."""
    import torch

    wm = agent._wm
    dyn = wm.dynamics
    swap = lambda x: x.permute([1, 0] + list(range(2, len(x.shape))))
    with torch.no_grad():
        data = wm.preprocess(batch)
        embed = wm.encoder(data)
        embed_t, action_t, isfirst_t = swap(embed), swap(data["action"]), swap(data["is_first"])
        state, feats = None, []
        for t in range(embed_t.shape[0]):  # mirror RSSM.observe's static_scan, sample=False
            post, _ = dyn.obs_step(state, action_t[t], embed_t[t], isfirst_t[t], sample=False)
            state = post
            feats.append(dyn.get_feat(post))
        feat = torch.stack(feats, 1)  # (B, T, F)
    return feat


def _actor_margin(agent, feat, a_star):
    """Mean argmax-margin + flip of the CURRENT actor at latents `feat` against the FIXED
    competent actions a_star. margin = logp(a*) - max_{a!=a*} logp(a). feat [B,T,F], a_star [B,T]."""
    import torch

    with torch.no_grad():
        dist = agent._task_behavior.actor(feat)
        logp = torch.log(dist.probs + 1e-9)
        lp_star = logp.gather(-1, a_star.unsqueeze(-1)).squeeze(-1)
        masked = logp.clone()
        masked.scatter_(-1, a_star.unsqueeze(-1), float("-inf"))
        lp_other = masked.max(-1).values
        margin = lp_star - lp_other
        flip = (dist.probs.argmax(-1) != a_star).float()
    return float(margin.mean()), float(flip.mean())


def measure_wm_on_A(agent, batch):
    """Current WM recon loss on a held A-batch via a no-grad forward (mirrors WorldModel._train
    line 140 without the optimizer step -- the model is NOT mutated)."""
    import torch

    wm = agent._wm
    with torch.no_grad():
        data = wm.preprocess(batch)
        embed = wm.encoder(data)
        post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
        feat = wm.dynamics.get_feat(post)
        pred = wm.heads["decoder"](feat)
        recon = -pred["image"].log_prob(data["image"]).mean()
    return float(recon)


class MarginLogger:
    """FROZEN-vs-LIVE margin disambiguator for actor-side vs representation-side forgetting.

    Holds ONE fixed A-batch (sequences) captured at A-competence, the competent latents, and the
    competent argmax a*. Each call measures the CURRENT actor's margin two ways:
      frozen_margin = current actor on the CACHED competent latents (actor drift on GOOD latents)
      live_margin   = current actor on latents RE-ENCODED through the CURRENT WM (the latents the
                      post-B WM actually produces for A)
    READ:
      frozen HOLDS + live COLLAPSES -> representation-side (WM feeds the good actor bad latents).
      frozen ALSO drops             -> actor co-degrades (not cleanly representation-side).
    wm_on_A_recon (unnormalized recon SSE) is kept as a SECONDARY proxy only -- it conflates
    'never learned' with 'forgot' and its ratio is inflated by a near-zero memorization baseline,
    so it does NOT decide Q2 (see STAGE_AB_LOCATE_RESULT_2026-06-21.md).

    VALIDATION GATE (the lambda=0==baseline check): at A-competence the live re-encode uses the
    SAME WM as the cached frozen latents, so live_margin MUST equal frozen_margin. Checked at
    construction (`gate_ok`) and again operationally in the first B-step-0 eval row. If they
    diverge, the re-encode path is mis-wired -> do NOT trust the run.
    """

    def __init__(self, agent, dataset, device="cuda"):
        import torch

        self.agent = agent
        self.device = device
        self.A_batch = next(dataset)  # raw held A sequences; always preprocessed via the WM
        self.feats_frozen = _encode_feats(agent, self.A_batch)  # competent latents, cached
        with torch.no_grad():
            self.a_star = agent._task_behavior.actor(self.feats_frozen).probs.argmax(-1)  # [B,T]
        # construction-time gate: the live re-encode path must reproduce the cached frozen result
        fm, _ = _actor_margin(agent, self.feats_frozen, self.a_star)
        lm, _ = _actor_margin(agent, _encode_feats(agent, self.A_batch), self.a_star)
        self.baseline = {"frozen_margin": fm, "live_margin": lm}
        self.gate_ok = abs(fm - lm) < 1e-3

    def __call__(self):
        fm, ff = _actor_margin(self.agent, self.feats_frozen, self.a_star)
        lm, lf = _actor_margin(self.agent, _encode_feats(self.agent, self.A_batch), self.a_star)
        return {
            "frozen_margin": fm, "frozen_flip": ff,
            "live_margin": lm, "live_flip": lf,
            "wm_on_A_recon": measure_wm_on_A(self.agent, self.A_batch),
            "gate_ok": bool(self.gate_ok),
        }


# --------------------------------------------------------------------------------------
# Self-test: wiring sanity on the competent ckpt (the real Q2 gate is the A->B run)
# --------------------------------------------------------------------------------------
def self_test(config_names, logdir, device, n, max_steps):
    print(f"[self_test] building agent from {logdir} ...")
    config = build_config(config_names, logdir, device)
    agent, _env_maker, dataset = load_agent(config, device)
    print(f"[self_test] loaded latest.pt | task={config.task} | num_actions={config.num_actions}")

    logger = MarginLogger(agent, dataset, device=device)
    row = logger()
    print("[self_test] metrics @ competent ckpt:")
    for k, v in row.items():
        print(f"    {k:>16}: {v}")

    # At the competent ckpt: frozen_margin>0 (actor prefers its own a*), and -- the GATE -- the
    # live re-encode through the SAME WM must reproduce the frozen result. wm recon finite.
    c1 = row["frozen_margin"] > 0.0
    c2 = row["frozen_flip"] < 0.05
    c_gate = abs(row["frozen_margin"] - row["live_margin"]) < 1e-3 and row["gate_ok"]
    c3 = (row["wm_on_A_recon"] == row["wm_on_A_recon"]) and abs(row["wm_on_A_recon"]) < 1e9
    print("\n=== SELF-TEST (wiring + re-encode VALIDATION GATE) ===")
    print(f"  (1) frozen_margin>0 at a*   : {row['frozen_margin']:+.3f}    -> {'PASS' if c1 else 'FAIL'}")
    print(f"  (2) frozen_flip~0 at a*     : {row['frozen_flip']:.3f}     -> {'PASS' if c2 else 'FAIL'}")
    print(f"  (GATE) live==frozen @comp   : frozen {row['frozen_margin']:+.3f} vs live {row['live_margin']:+.3f}"
          f"  -> {'PASS' if c_gate else 'FAIL'}")
    print(f"  (3) wm_on_A finite (proxy)  : {row['wm_on_A_recon']:.4f}  -> {'PASS' if c3 else 'FAIL'}")
    if c1 and c2 and c_gate and c3:
        print("  -> ALL PASS: re-encode path validated (live reproduces frozen at competence).")
        print("     Q2 disambiguation is LIVE in the A->B run: frozen holds + live collapses = rep-side;")
        print("     both drop = actor co-degrades. Ready to attach to the orchestrator.")
    else:
        print("  -> FAIL: re-encode/wiring bug (preprocess drift, encoder/observe mismatch). Fix before the run.")
    return row


def _parser():
    p = argparse.ArgumentParser(description="NM512 A->B Q2 margin/WM-on-A logger + self-test.")
    p.add_argument("--self_test", action="store_true")
    p.add_argument("--logdir", required=True)
    p.add_argument("--configs", nargs="+", default=["minigrid"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--max_steps", type=int, default=256)
    return p


def main(argv=None):
    a = _parser().parse_args(argv)
    if a.self_test:
        self_test(a.configs, a.logdir, a.device, a.n, a.max_steps)
    else:
        print("nothing to do; pass --self_test or import MarginLogger from the orchestrator.")


if __name__ == "__main__":
    main()
