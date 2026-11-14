"""Warm-start hook: load ONLY the actor from a deployed-policy checkpoint, then
warm up the critic before fine-tuning the actor.

mjlab's `train --agent.resume` performs a FULL resume (actor + critic + optimizer +
iteration). To warm-start from the official deployed policy we want just the actor
weights, leaving the critic, optimizer, and iteration counter fresh. rsl_rl's
``PPO.load`` already supports this via ``load_cfg``; this hook detects a checkpoint
whose ``infos`` carries ``warm_start_actor_only=True`` and forces an actor-only load.

CRITICAL: an actor-only warm-start with a *fresh* critic collapses — PPO's first
update uses noise advantages from the random critic and destroys the good actor.
To prevent this, the hook also freezes the actor for the first
``infos["critic_warmup_iters"]`` iterations so the critic learns the value function
from the (good) official-policy rollouts, then unfreezes the actor to fine-tune with
real advantages. rsl_rl calls ``PPO.update()`` once per learning iteration, so the
warm-up counter is driven from a patched ``PPO.update``.

The decisions are pure functions (``warm_start_load_cfg``, ``critic_warmup_iters``)
so they are unit-testable without importing mjlab/torch-heavy runner code.
"""
from __future__ import annotations

ACTOR_ONLY_LOAD_CFG = {
    "actor": True,
    "critic": False,
    "optimizer": False,
    "iteration": False,
    "rnd": False,
}


def warm_start_load_cfg(loaded_dict: dict, load_cfg: dict | None) -> dict | None:
    """Return an actor-only load_cfg for a warm-start checkpoint, else pass through.

    Only overrides when the caller did not specify an explicit load_cfg (the normal
    `--agent.resume` path) AND the checkpoint is marked warm_start_actor_only.
    """
    if load_cfg is None and bool(loaded_dict.get("infos", {}).get("warm_start_actor_only")):
        return dict(ACTOR_ONLY_LOAD_CFG)
    return load_cfg


def critic_warmup_iters(loaded_dict: dict) -> int:
    """Number of critic-warmup iterations requested by a warm-start checkpoint.

    Returns 0 for non-warm-start checkpoints or when no warm-up is requested.
    """
    infos = loaded_dict.get("infos", {})
    if not bool(infos.get("warm_start_actor_only")):
        return 0
    try:
        return max(0, int(infos.get("critic_warmup_iters", 0)))
    except (TypeError, ValueError):
        return 0


def install() -> None:
    """Patch runner load + PPO.update for warm-start + critic warm-up (idempotent)."""
    from mjlab.rl.runner import MjlabOnPolicyRunner

    if getattr(MjlabOnPolicyRunner, "_warm_start_hook_installed", False):
        return
    import torch
    from rsl_rl.algorithms import PPO

    _orig_load = MjlabOnPolicyRunner.load

    def _load(self, path, load_cfg=None, strict=True, map_location=None):
        try:
            probe = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            probe = {}
        resolved = warm_start_load_cfg(probe, load_cfg)
        if load_cfg is None and resolved == ACTOR_ONLY_LOAD_CFG:
            print(f"[warm-start] actor-only load from {path} (critic/optimizer/iter fresh)")
            n_warm = critic_warmup_iters(probe)
            if n_warm > 0:
                self.alg._critic_warmup_iters = n_warm
                self.alg._critic_warmup_count = 0
                self.alg.actor.requires_grad_(False)
                print(f"[warm-start] critic warm-up: actor frozen for first {n_warm} iters")
        return _orig_load(self, path, load_cfg=resolved, strict=strict, map_location=map_location)

    MjlabOnPolicyRunner.load = _load

    _orig_update = PPO.update

    def _update(self):
        n_warm = getattr(self, "_critic_warmup_iters", 0)
        if n_warm <= 0:
            return _orig_update(self)
        loss = _orig_update(self)  # actor frozen -> only critic updates
        cnt = getattr(self, "_critic_warmup_count", 0) + 1
        self._critic_warmup_count = cnt
        if cnt >= n_warm:
            self.actor.requires_grad_(True)
            self._critic_warmup_iters = 0
            print(f"[warm-start] critic warm-up complete ({cnt} iters); actor unfrozen")
        return loss

    PPO.update = _update
    MjlabOnPolicyRunner._warm_start_hook_installed = True
