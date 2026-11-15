"""Run the same seeded starts on each foot, surface, and policy. Python compare.py --help"""

import argparse
import json
from pathlib import Path

from duck_env import DuckSitStandEnv
from lab import evaluate, policy
from learner import SOURCE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=811)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    models = {"factory": policy(SOURCE)}
    if args.candidate:
        models["trained"] = policy(args.candidate)
    results = []
    for foot in ("v2-stock", "v2-heel15"):
        for surface in ("hard", "carpet"):
            env = DuckSitStandEnv(foot, surface)
            for label, predict in models.items():
                for start in ("sit", "stand"):
                    r = evaluate(env, predict, range(args.seed, args.seed + args.episodes), start)
                    results.append(dict(policy=label, **r))
                    print(
                        f"{label} {foot} {surface} {start}: {r['successes']}/{r['episodes']}, {r['falls']} falls",
                        flush=True,
                    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
