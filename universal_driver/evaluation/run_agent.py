"""Runtime entry-point to evaluate a trained agent in CARLA."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

from ..config import AgentConfig, DEFAULT_AGENT_CONFIG
from ..training.train import CarlaEnvAdapter, Trainer


def load_checkpoint(path: Path) -> Dict:
    return torch.load(path, map_location="cpu")


def build_agent(config: AgentConfig) -> Trainer:
    env = CarlaEnvAdapter()
    trainer = Trainer(env, config)
    return trainer


def run(checkpoint_path: Path, steps: int, config: AgentConfig) -> None:
    checkpoint = load_checkpoint(checkpoint_path)
    loaded_config: AgentConfig = checkpoint.get("config", config)
    trainer = build_agent(loaded_config)
    trainer.policy.load_state_dict(checkpoint["policy"])
    trainer.perception.load_state_dict(checkpoint["perception"])
    trainer.rollout(steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained Universal Driver agent")
    parser.add_argument("checkpoint", type=Path, help="Path to the saved checkpoint")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    run(args.checkpoint, steps=args.steps, config=DEFAULT_AGENT_CONFIG)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()

