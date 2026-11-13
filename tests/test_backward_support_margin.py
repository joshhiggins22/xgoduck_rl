"""Standalone unit tests for ``backward_support_margin`` (carpet sit/stand).

Runs WITHOUT mjlab / GPU / a training venv: mdp.py imports mjlab at module
level, so when that import is unavailable the function source is extracted
from mdp.py via AST and exec'd against mocked asset/command tensors. In a
full training venv the real module import is used instead — same tests.

Math under test (yaw = 0 → forward = +x, feet sites at x = 0):
    heel_x  = -heel_offset                     (default 0.0329 m, measured)
    margin  = com_x - heel_x = com_x + 0.0329
    penalty = clamp(safe_margin - margin, 0)   (default safe_margin 0.005 m)
Gated ON only when STAND commanded (twist cmd col 0 < 0.5) AND trunk z
(relative to the terrain origin) > height_gate_low.

Run:  python tests/test_backward_support_margin.py   (or pytest)
"""

import ast
import inspect
import math
import pathlib
import types

import torch

_TASKS_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "mjlab_microduck" / "tasks"
_MDP_PATH = _TASKS_DIR / "mdp.py"
_INIT_PATH = _TASKS_DIR / "__init__.py"

# XGODUCK_SIT_Z + 0.02 (xgoduck_constants.XGODUCK_SIT_Z = 0.085868).
_GATE_LOW = 0.085868 + 0.02
_HEEL = 0.0329   # measured heel_offset default
_SAFE = 0.005    # safe_margin default


class _StubSceneEntityCfg:
    def __init__(self, name, site_names=None):
        self.name = name
        self.site_names = site_names
        self.site_ids = [0, 1]


def _load_reward():
    """Real import when mjlab is installed; AST extraction otherwise."""
    try:
        from mjlab_microduck.tasks.mdp import backward_support_margin
        return backward_support_margin
    except Exception:
        tree = ast.parse(_MDP_PATH.read_text())
        func = next(
            n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "backward_support_margin"
        )
        module = ast.Module(body=[func], type_ignores=[])
        ast.fix_missing_locations(module)
        ns = {
            "torch": torch,
            "SceneEntityCfg": _StubSceneEntityCfg,
            "_DEFAULT_ASSET_CFG": _StubSceneEntityCfg("robot"),
            "ManagerBasedRlEnv": object,
            "Entity": object,
        }
        exec(compile(module, str(_MDP_PATH), "exec"), ns)
        return ns["backward_support_margin"]


backward_support_margin = _load_reward()


# ── Mocks ────────────────────────────────────────────────────────────────────

class _Data:
    def __init__(self, com, quat, pos, sites):
        self.root_com_pos_w = torch.tensor(com, dtype=torch.float32)
        self.root_link_quat_w = torch.tensor(quat, dtype=torch.float32)
        self.root_link_pos_w = torch.tensor(pos, dtype=torch.float32)
        self.site_pos_w = torch.tensor(sites, dtype=torch.float32)


class _Asset:
    def __init__(self, data, with_find_sites=True):
        self.data = data
        if with_find_sites:
            self.find_sites = self._find_sites

    @staticmethod
    def _find_sites(names, preserve_order=True):
        assert list(names) == ["left_foot", "right_foot"]
        assert preserve_order is True
        return [0, 1], list(names)


class _Scene(dict):
    def __init__(self, asset, origins):
        super().__init__(robot=asset)
        self.terrain = types.SimpleNamespace(
            env_origins=torch.tensor(origins, dtype=torch.float32)
        )


class _CommandManager:
    def __init__(self, cmd):
        self._cmd = torch.tensor(cmd, dtype=torch.float32)

    def get_command(self, name):
        assert name == "twist"
        return self._cmd


class _Env:
    def __init__(self, asset, cmd, origins):
        self.scene = _Scene(asset, origins)
        self.command_manager = _CommandManager(cmd)


def _make_env(com_xy, yaw=0.0, z=0.14, sit_flag=0.0, foot_x=0.0, origin_z=0.0,
              with_find_sites=True):
    """One env: trunk at origin yaw/z, feet sites at (foot_x, ±0.03, 0)."""
    com_xy = list(com_xy)
    quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    data = _Data(
        com=[com_xy + [z]],
        quat=[quat],
        pos=[[0.0, 0.0, z]],
        sites=[[[foot_x, -0.03, 0.0], [foot_x, 0.03, 0.0]]],
    )
    return _Env(
        _Asset(data, with_find_sites=with_find_sites),
        cmd=[[sit_flag, 0.0, 0.0]],
        origins=[[0.0, 0.0, origin_z]],
    )


def _penalty(env, **kwargs):
    out = backward_support_margin(env, height_gate_low=_GATE_LOW, **kwargs)
    assert out.shape == (1,)
    return float(out[0])


# ── Reward math ──────────────────────────────────────────────────────────────

def test_zero_when_com_well_ahead_of_heel():
    # margin = 0.05 + 0.0329 = 0.0829 m >> safe_margin → no penalty.
    assert _penalty(_make_env([0.05, 0.0])) == 0.0


def test_positive_when_com_behind_heel():
    # margin = -0.04 + 0.0329 = -0.0071 → penalty = 0.005 + 0.0071 = 0.0121.
    p = _penalty(_make_env([-0.04, 0.0]))
    assert p > 0.0
    assert abs(p - 0.0121) < 1e-6


def test_zero_when_sit_commanded():
    # Same unsafe COM, but SIT flag → gate off.
    assert _penalty(_make_env([-0.04, 0.0], sit_flag=1.0)) == 0.0


def test_zero_when_trunk_below_height_gate():
    # STAND commanded but trunk below height_gate_low (seated regime) → 0.
    assert _penalty(_make_env([-0.04, 0.0], z=_GATE_LOW - 0.01)) == 0.0
    # Exactly at the gate is still OFF (strict >).
    assert _penalty(_make_env([-0.04, 0.0], z=_GATE_LOW)) == 0.0


def test_known_margin_magnitude():
    # margin = 0.002 m (inside safe band) → penalty = 0.005 - 0.002 = 0.003.
    p = _penalty(_make_env([-_HEEL + 0.002, 0.0]))
    assert abs(p - 0.003) < 1e-6


def test_zero_at_safe_margin_boundary():
    # margin == safe_margin → clamp(0) = 0 (soft wall starts beyond it).
    assert _penalty(_make_env([-_HEEL + _SAFE, 0.0])) == 0.0


def test_no_attempt_tax_on_forward_leaning_rise():
    # Normal rise: COM ahead of the heel, trunk above the gate → exactly 0.
    for com_x in (0.0, 0.01, 0.02, 0.05):
        assert _penalty(_make_env([com_x, 0.0], z=0.12)) == 0.0


def test_yaw_rotated_forward():
    # yaw = +90° → forward = +y; COM 7.1 mm behind the heel along y.
    p = _penalty(_make_env([0.0, -0.04], yaw=math.pi / 2))
    assert abs(p - 0.0121) < 1e-6
    # Same COM offset along x is now LATERAL → margin 0.0329 → no penalty.
    assert _penalty(_make_env([-0.04, 0.0], yaw=math.pi / 2)) == 0.0


def test_penalty_never_negative_and_grows_backward():
    ps = [_penalty(_make_env([x, 0.0])) for x in (0.05, 0.0, -0.0279, -0.04, -0.2)]
    assert all(p >= 0.0 for p in ps)
    assert ps == sorted(ps)  # monotone as the COM drifts back
    assert abs(ps[-1] - (_SAFE + 0.2 - _HEEL)) < 1e-6


def test_height_gate_uses_terrain_origin():
    # z=0.11 with origin_z=0.01 → relative z=0.10 < gate → 0 (rough terrain).
    assert _penalty(_make_env([-0.04, 0.0], z=0.11, origin_z=0.01)) == 0.0
    assert _penalty(_make_env([-0.04, 0.0], z=0.11, origin_z=0.0)) > 0.0


def test_params_override():
    # Larger heel_offset moves the wall back: margin = -0.04 + 0.05 = 0.01 → 0.
    assert _penalty(_make_env([-0.04, 0.0]), heel_offset=0.05) == 0.0
    # Larger safe_margin widens the penalized band.
    p = _penalty(_make_env([0.0, 0.0]), safe_margin=0.05)
    assert abs(p - (0.05 - _HEEL)) < 1e-6


def test_batch_mixed():
    env = _make_env([0.05, 0.0])
    # Extend to 2 envs: safe + unsafe.
    d = env.scene["robot"].data
    d.root_com_pos_w = torch.tensor([[0.05, 0.0, 0.14], [-0.04, 0.0, 0.14]])
    d.root_link_quat_w = d.root_link_quat_w.repeat(2, 1)
    d.root_link_pos_w = d.root_link_pos_w.repeat(2, 1)
    d.site_pos_w = d.site_pos_w.repeat(2, 1, 1)
    env.command_manager._cmd = env.command_manager._cmd.repeat(2, 1)
    env.scene.terrain.env_origins = env.scene.terrain.env_origins.repeat(2, 1)
    out = backward_support_margin(env, height_gate_low=_GATE_LOW)
    assert out.shape == (2,)
    assert float(out[0]) == 0.0
    assert abs(float(out[1]) - 0.0121) < 1e-6


def test_site_ids_fallback_without_find_sites():
    # mjlab builds lacking Entity.find_sites use manager-resolved site_ids.
    env = _make_env([-0.04, 0.0], with_find_sites=False)
    cfg = _StubSceneEntityCfg("robot", site_names=("left_foot", "right_foot"))
    out = backward_support_margin(env, height_gate_low=_GATE_LOW, asset_cfg=cfg)
    assert abs(float(out[0]) - 0.0121) < 1e-6


def test_signature_defaults():
    sig = inspect.signature(backward_support_margin)
    assert sig.parameters["safe_margin"].default == 0.005
    assert sig.parameters["heel_offset"].default == 0.0329
    assert sig.parameters["command_name"].default == "twist"
    assert sig.parameters["height_gate_low"].default is inspect.Parameter.empty


# ── Static registration well-formedness (no mjlab import needed) ─────────────

def test_carpet_registration_static():
    tree = ast.parse(_INIT_PATH.read_text())

    # Factory exists and builds on the rough sitstand base.
    factory = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_xgoduck_sitstand_carpet_cfg"
    )
    base_calls = [
        n for n in ast.walk(factory)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "_xgoduck_sitstand_cfg"
    ]
    assert base_calls, "carpet factory must call _xgoduck_sitstand_cfg"
    assert any(
        any(kw.arg == "rough" and getattr(kw.value, "value", None) is True
            for kw in c.keywords)
        for c in base_calls
    ), "carpet factory must pass rough=True"

    # Reward term: negative weight + required params.
    reward_assigns = [
        n for n in ast.walk(factory)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Subscript)
        and getattr(n.targets[0].slice, "value", None) == "backward_support_margin"
    ]
    assert len(reward_assigns) == 1
    call = reward_assigns[0].value
    assert isinstance(call, ast.Call) and call.func.id == "RewardTermCfg"
    kws = {kw.arg: kw.value for kw in call.keywords}
    weight = kws["weight"]
    weight_val = -weight.operand.value if isinstance(weight, ast.UnaryOp) else weight.value
    assert weight_val == -2.0, "backward_support_margin must have a NEGATIVE weight"
    param_keys = {k.value for k in kws["params"].keys}
    assert {"asset_cfg", "safe_margin", "heel_offset", "height_gate_low",
            "command_name"} <= param_keys

    # Compliance DR event registered in reset mode.
    event_assigns = [
        n for n in ast.walk(factory)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Subscript)
        and getattr(n.targets[0].slice, "value", None) == "carpet_compliance"
    ]
    assert len(event_assigns) == 1
    ekws = {kw.arg: kw.value for kw in event_assigns[0].value.keywords}
    assert ekws["mode"].value == "reset"

    # Task + play registration with the shared sitstand rl cfg / runner.
    regs = [
        n for n in tree.body
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "register_mjlab_task"
    ]
    by_id = {}
    for r in regs:
        rkws = {kw.arg: kw.value for kw in r.value.keywords}
        tid = rkws["task_id"].value
        by_id[tid] = rkws
    carpet = by_id["Mjlab-SitStand-Carpet-XgoDuck"]
    assert carpet["env_cfg"].func.id == "_xgoduck_sitstand_carpet_cfg"
    play = carpet["play_env_cfg"]
    assert play.func.id == "_xgoduck_sitstand_carpet_cfg"
    assert any(kw.arg == "play" and kw.value.value is True for kw in play.keywords)
    assert carpet["rl_cfg"].id == "XgoduckSitStandRlCfg"
    assert carpet["runner_cls"].id == "MicroduckOnPolicyRunner"

    # Baseline registrations untouched.
    assert "Mjlab-SitStand-Flat-XgoDuck" in by_id
    assert "Mjlab-SitStand-Rough-XgoDuck" in by_id
    assert by_id["Mjlab-SitStand-Flat-XgoDuck"]["env_cfg"].func.id == "_xgoduck_sitstand_cfg"
    assert by_id["Mjlab-SitStand-Rough-XgoDuck"]["env_cfg"].func.id == "_xgoduck_sitstand_cfg"


# ── Carpet-compliance DR event (realistic raw-MjModel mock) ─────────────────
# Regression: the raw mujoco.MjModel has NO .body_names attribute. The mock
# below deliberately omits it (names are only reachable via mj_id2name), so a
# revert to model.body_names raises AttributeError and fails these tests.

class _MockMujoco:
    class mjtObj:
        mjOBJ_BODY = 1

    class mjtGeom:
        mjGEOM_PLANE = 0
        mjGEOM_HFIELD = 1
        mjGEOM_MESH = 2

    @staticmethod
    def mj_id2name(model, objtype, i):
        assert objtype == _MockMujoco.mjtObj.mjOBJ_BODY
        return model._names[i]


class _RawMjModel:
    """Mimics mujoco.MjModel: no .body_names; names only via mj_id2name."""

    def __init__(self):
        self.nbody = 4
        self.ngeom = 5
        self._names = ["world", "terrain", "trunk_base", "left_ankle"]
        # g0 terrain hfield, g1 floor plane, g2/g3 robot meshes, g4 visual-only.
        self.geom_contype = [1, 1, 1, 1, 0]
        self.geom_conaffinity = [1, 1, 1, 1, 0]
        self.geom_bodyid = [1, 0, 2, 3, 2]
        self.geom_type = [1, 0, 2, 2, 2]


class _DRSimModel:
    def __init__(self, nworld, ngeom):
        self.geom_solref = torch.zeros(nworld, ngeom, 2)


class _DREnv:
    def __init__(self, nworld=3):
        self.num_envs = nworld
        self.device = "cpu"
        mj = _RawMjModel()
        self.sim = types.SimpleNamespace(
            mj_model=mj, model=_DRSimModel(nworld, mj.ngeom)
        )


def _load_dr():
    try:
        from mjlab_microduck.tasks.mdp import randomize_ground_compliance
        return randomize_ground_compliance
    except Exception:
        tree = ast.parse(_MDP_PATH.read_text())
        func = next(
            n for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "randomize_ground_compliance"
        )
        func.decorator_list = []  # @requires_model_fields is an mjlab marker
        module = ast.Module(body=[func], type_ignores=[])
        ast.fix_missing_locations(module)
        ns = {"torch": torch, "mujoco": _MockMujoco, "ManagerBasedRlEnv": object}
        exec(compile(module, str(_MDP_PATH), "exec"), ns)
        return ns["randomize_ground_compliance"]


randomize_ground_compliance = _load_dr()


def test_dr_modifies_only_ground_geoms():
    env = _DREnv(nworld=3)
    randomize_ground_compliance(env, torch.arange(3))
    sr = env.sim.model.geom_solref
    assert torch.all(sr[:, 0, 0] >= 0.02) and torch.all(sr[:, 0, 0] <= 0.06)
    assert torch.all(sr[:, 1, 0] >= 0.02) and torch.all(sr[:, 1, 0] <= 0.06)
    # Robot meshes (2,3) and visual-only geom (4) untouched.
    assert torch.all(sr[:, 2:, :] == 0.0)


def test_dr_per_env_uniform_and_damping():
    env = _DREnv(nworld=4)
    randomize_ground_compliance(env, torch.arange(4), damping_ratio=1.0)
    sr = env.sim.model.geom_solref
    assert torch.all(sr[:, 0, 0] == sr[:, 1, 0])  # one stiffness per env
    assert torch.all(sr[:, 0, 1] == 1.0) and torch.all(sr[:, 1, 1] == 1.0)


def test_dr_non_accumulating():
    env = _DREnv(nworld=2)
    randomize_ground_compliance(env, torch.arange(2))
    randomize_ground_compliance(env, torch.arange(2))  # second reset
    sr = env.sim.model.geom_solref
    assert torch.all(sr[:, 0, 0] >= 0.02) and torch.all(sr[:, 0, 0] <= 0.06)


def test_dr_empty_env_ids_noop():
    env = _DREnv(nworld=2)
    randomize_ground_compliance(env, torch.tensor([], dtype=torch.long))
    assert torch.all(env.sim.model.geom_solref == 0.0)


def test_dr_no_ground_geoms_raises():
    env = _DREnv(nworld=1)
    mj = env.sim.mj_model
    mj.geom_type = [2, 2, 2, 2, 2]       # all meshes
    mj.geom_bodyid = [2, 2, 2, 3, 2]     # all under robot bodies
    mj._names = ["world", "notterrain", "trunk_base", "left_ankle"]
    try:
        randomize_ground_compliance(env, torch.arange(1))
    except ValueError:
        return
    raise AssertionError("expected ValueError when no ground geoms exist")


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    raise SystemExit(1 if failed else 0)
