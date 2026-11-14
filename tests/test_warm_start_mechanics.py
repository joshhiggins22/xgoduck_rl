"""Torch-based mechanics test for the critic warm-up freeze/unfreeze.

Validates the core claim behind the warm-up: with the actor frozen
(requires_grad=False, .grad cleared) a single Adam optimizer over actor+critic
leaves the actor bit-identical while the critic updates; after unfreezing, the
actor updates. Skipped where torch is unavailable (runs on the training host).
"""
import unittest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch required")
class FreezeMechanicsTest(unittest.TestCase):
    def _make(self):
        torch.manual_seed(0)
        actor = torch.nn.Linear(4, 4)
        critic = torch.nn.Linear(4, 4)
        opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-2)
        return actor, critic, opt

    def test_frozen_actor_bit_identical_critic_updates(self):
        actor, critic, opt = self._make()
        actor.requires_grad_(False)
        for p in actor.parameters():
            p.grad = None
        actor_before = [p.clone() for p in actor.parameters()]
        critic_before = [p.clone() for p in critic.parameters()]
        x = torch.randn(8, 4)
        for _ in range(5):
            opt.zero_grad()
            critic(x).pow(2).mean().backward()  # no grad path to frozen actor
            opt.step()
        for a, b in zip(actor.parameters(), actor_before):
            self.assertTrue(torch.equal(a, b), "frozen actor changed during warm-up")
        self.assertTrue(
            any(not torch.equal(a, b) for a, b in zip(critic.parameters(), critic_before)),
            "critic did not update during warm-up",
        )

    def test_unfrozen_actor_updates(self):
        actor, critic, opt = self._make()
        actor.requires_grad_(False)
        x = torch.randn(8, 4)
        for _ in range(3):  # warm-up
            opt.zero_grad(); critic(x).pow(2).mean().backward(); opt.step()
        actor.requires_grad_(True)  # unfreeze
        before = [p.clone() for p in actor.parameters()]
        for _ in range(3):
            opt.zero_grad()
            (actor(x).pow(2).mean() + critic(x).pow(2).mean()).backward()
            opt.step()
        self.assertTrue(
            any(not torch.equal(a, b) for a, b in zip(actor.parameters(), before)),
            "unfrozen actor did not update",
        )


if __name__ == "__main__":
    unittest.main()
