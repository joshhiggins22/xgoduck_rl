"""Unit tests for the warm-start actor-only load decision (no mjlab import needed)."""
import unittest

from mjlab_microduck.warm_start_hook import ACTOR_ONLY_LOAD_CFG, warm_start_load_cfg


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


if __name__ == "__main__":
    unittest.main()
