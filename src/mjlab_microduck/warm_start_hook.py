"""Warm-start hook: load ONLY the actor from a deployed-policy checkpoint.

mjlab's `train --agent.resume` performs a FULL resume (actor + critic + optimizer +
iteration). To warm-start from the official deployed policy we want just the actor
weights, leaving the critic, optimizer, and iteration counter fresh. rsl_rl's
``PPO.load`` already supports this via ``load_cfg``; this hook detects a checkpoint
whose ``infos`` carries ``warm_start_actor_only=True`` and forces an actor-only load.

The decision is a pure function (``warm_start_load_cfg``) so it is unit-testable
without importing mjlab/torch-heavy runner code.
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


def install() -> None:
    """Patch MjlabOnPolicyRunner.load to honor warm-start checkpoints (idempotent)."""
    from mjlab.rl.runner import MjlabOnPolicyRunner

    if getattr(MjlabOnPolicyRunner, "_warm_start_hook_installed", False):
        return
    import torch

    _orig_load = MjlabOnPolicyRunner.load

    def _load(self, path, load_cfg=None, strict=True, map_location=None):
        try:
            probe = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            probe = {}
        resolved = warm_start_load_cfg(probe, load_cfg)
        if resolved is not load_cfg and resolved == ACTOR_ONLY_LOAD_CFG:
            print(f"[warm-start] actor-only load from {path} (critic/optimizer/iter fresh)")
        return _orig_load(self, path, load_cfg=resolved, strict=strict, map_location=map_location)

    MjlabOnPolicyRunner.load = _load
    MjlabOnPolicyRunner._warm_start_hook_installed = True
