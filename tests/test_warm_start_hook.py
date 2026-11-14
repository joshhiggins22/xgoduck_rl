"""Unit tests for the warm-start actor-only load decision (no mjlab import needed)."""
import unittest

from mjlab_microduck.warm_start_hook import ACTOR_ONLY_LOAD_CFG, critic_warmup_iters, warm_start_load_cfg


class WarmStartLoadCfgTest(unittest.TestCase):
    def test_warmstart_checkpoint_forces_actor_only(self):
        ck = {"infos": {"warm_start_actor_only": True}}
        self.assertEqual(warm_start_load_cfg(ck, None), ACTOR_ONLY_LOAD_CFG)

    def test_normal_checkpoint_full_resume_unchanged(self):
        ck = {"infos": {"env_state": {"common_step_counter": 100}}}
        self.assertIsNone(warm_start_load_cfg(ck, None))

    def test_explicit_load_cfg_is_respected(self):
        ck = {"infos": {"warm_start_actor_only": True}}
        explicit = {"actor": True, "critic": True, "optimizer": True, "iteration": True}
        self.assertEqual(warm_start_load_cfg(ck, explicit), explicit)

    def test_missing_infos_is_full_resume(self):
        self.assertIsNone(warm_start_load_cfg({}, None))

    def test_flag_false_is_full_resume(self):
        ck = {"infos": {"warm_start_actor_only": False}}
        self.assertIsNone(warm_start_load_cfg(ck, None))

    def test_actor_only_cfg_leaves_critic_optimizer_iter_fresh(self):
        cfg = warm_start_load_cfg({"infos": {"warm_start_actor_only": True}}, None)
        assert cfg is not None
        self.assertTrue(cfg["actor"])
        self.assertFalse(cfg["critic"])
        self.assertFalse(cfg["optimizer"])
        self.assertFalse(cfg["iteration"])


class CriticWarmupItersTest(unittest.TestCase):
    def test_warmstart_with_warmup(self):
        ck = {"infos": {"warm_start_actor_only": True, "critic_warmup_iters": 50}}
        self.assertEqual(critic_warmup_iters(ck), 50)

    def test_warmstart_without_warmup_defaults_zero(self):
        self.assertEqual(critic_warmup_iters({"infos": {"warm_start_actor_only": True}}), 0)

    def test_non_warmstart_returns_zero(self):
        self.assertEqual(critic_warmup_iters({"infos": {"critic_warmup_iters": 50}}), 0)

    def test_invalid_value_returns_zero(self):
        ck = {"infos": {"warm_start_actor_only": True, "critic_warmup_iters": "abc"}}
        self.assertEqual(critic_warmup_iters(ck), 0)

    def test_negative_clamped_to_zero(self):
        ck = {"infos": {"warm_start_actor_only": True, "critic_warmup_iters": -5}}
        self.assertEqual(critic_warmup_iters(ck), 0)


if __name__ == "__main__":
    unittest.main()
