"""Training entry-point for the Universal Driver."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..config import AgentConfig, DEFAULT_AGENT_CONFIG
from ..control.controller import UniversalController
from ..perception.detector import MultiModalPerception
from ..planning.policy import UniversalPolicy
from ..sensors.base import Observation, SENSOR_REGISTRY
from ..supervisor import SafetySupervisor
from ..utils.logging import setup_logging
from ..utils.replay_buffer import ReplayBuffer, Transition

LOGGER = logging.getLogger(__name__)


class Environment:
    """Interface expected from driving simulators."""

    def reset(self) -> Dict[str, Observation]:  # pragma: no cover - interface
        raise NotImplementedError

    def step(self, command) -> Tuple[Dict[str, Observation], float, bool, Dict[str, float]]:  # pragma: no cover - interface
        raise NotImplementedError

    def get_speed(self) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def register_sensor(self, name: str, adapter) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class CarlaEnvAdapter(Environment):
    """Adapter wrapping the CARLA simulator."""

    def __init__(self, host: str = "127.0.0.1", port: int = 2000, timeout: float = 2.0):
        try:  # pragma: no cover - optional dependency
            import carla  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("carla package is required to use CarlaEnvAdapter") from exc
        self._client = carla.Client(host, port)
        self._client.set_timeout(timeout)
        self._world = self._client.get_world()
        self._vehicle: Optional[carla.Vehicle] = None
        self._sensors: Dict[str, Any] = {}
        self._latest_obs: Dict[str, Observation] = {}

    def reset(self) -> Dict[str, Observation]:  # pragma: no cover - requires CARLA
        import carla  # type: ignore

        self._destroy_sensors()
        if self._vehicle is not None:
            self._vehicle.destroy()
        blueprint = self._world.get_blueprint_library().filter("vehicle")[0]
        spawn_point = np.random.choice(self._world.get_map().get_spawn_points())
        self._vehicle = self._world.spawn_actor(blueprint, spawn_point)
        self._attach_sensors()
        self._latest_obs = {}
        return self.poll_sensors()

    def register_sensor(self, name: str, adapter) -> None:  # pragma: no cover - requires CARLA
        self._sensors[name] = adapter
        if self._vehicle is not None:
            adapter.initialize(self._world, parent=self._vehicle)

    def poll_sensors(self) -> Dict[str, Observation]:  # pragma: no cover - requires CARLA
        for name, adapter in self._sensors.items():
            self._latest_obs[name] = adapter.read()
        return self._latest_obs

    def step(self, command) -> Tuple[Dict[str, Observation], float, bool, Dict[str, float]]:  # pragma: no cover - requires CARLA
        assert self._vehicle is not None
        self._vehicle.apply_control(command)
        self._world.tick()
        obs = self.poll_sensors()
        reward = 0.0
        done = False
        info = {}
        return obs, reward, done, info

    def get_speed(self) -> float:  # pragma: no cover - requires CARLA
        if self._vehicle is None:
            return 0.0
        velocity = self._vehicle.get_velocity()
        return float(np.linalg.norm([velocity.x, velocity.y, velocity.z]))

    def _attach_sensors(self) -> None:  # pragma: no cover - requires CARLA
        for adapter in self._sensors.values():
            adapter.initialize(self._world, parent=self._vehicle)

    def _destroy_sensors(self) -> None:  # pragma: no cover - requires CARLA
        for adapter in self._sensors.values():
            try:
                adapter.destroy()
            except Exception:
                LOGGER.debug("Failed to destroy sensor %s", getattr(adapter, "config", None))
        self._latest_obs = {}


class Trainer:
    def __init__(
        self,
        env: Environment,
        config: AgentConfig = DEFAULT_AGENT_CONFIG,
        log_dir: Optional[Path] = None,
    ) -> None:
        self.env = env
        self.config = config
        setup_logging(log_dir)
        self.device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
        self.perception = MultiModalPerception(config.sensors, config.detector).to(self.device)
        self.policy = UniversalPolicy(256, config.policy).to(self.device)
        self.controller = UniversalController(config.vehicle)
        self.supervisor = SafetySupervisor(config.vehicle, config.supervisor)
        self.replay = ReplayBuffer(config.training.buffer_size)
        self.optimizer = torch.optim.Adam(
            list(self.perception.parameters()) + list(self.policy.parameters()),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.sensor_adapters = {
            sensor.name: SENSOR_REGISTRY.create(sensor) for sensor in config.sensors
        }
        for name, adapter in self.sensor_adapters.items():
            try:
                self.env.register_sensor(name, adapter)
            except NotImplementedError:
                LOGGER.debug("Environment does not support registering sensor %s", name)
            except AttributeError:
                LOGGER.debug("Environment missing register_sensor for %s", name)

    def normalize_observations(
        self, observations: Dict[str, Observation]
    ) -> Dict[str, Observation]:
        normalized: Dict[str, Observation] = {}
        for sensor in self.config.sensors:
            if sensor.name in observations:
                normalized[sensor.name] = observations[sensor.name]
            else:
                zeros = np.zeros(tuple(sensor.shape), dtype=np.float32)
                normalized[sensor.name] = Observation(
                    sensor=sensor,
                    data=zeros,
                    timestamp=0.0,
                )
        return normalized

    def collect_step(self, observation: Dict[str, Observation]) -> Tuple[Transition, Dict[str, Observation]]:
        normalized = self.normalize_observations(observation)
        perception_output = self.perception(normalized)
        features = perception_output.fused_features.to(self.device)
        policy_output = self.policy.act(features)
        action = policy_output.action.detach().cpu().numpy()
        command = self.controller.action_to_control(action)
        speed = getattr(self.env, "get_speed", lambda: 0.0)()
        decision = self.supervisor.enforce(command, perception_output, speed)
        raw_command = getattr(self.controller, "to_carla_control", lambda x: x)(decision.command)
        next_obs, reward, done, info = self.env.step(raw_command)
        next_normalized = self.normalize_observations(next_obs)
        next_perception = self.perception(next_normalized)
        transition = Transition(
            observation={"fused": features.detach().cpu().numpy()},
            action=action,
            reward=float(reward),
            next_observation={"fused": next_perception.fused_features.detach().cpu().numpy()},
            done=done,
            log_prob=float(policy_output.log_prob.item()),
            value=float(policy_output.value.item()),
        )
        info = dict(info)
        info.update({"overridden": float(decision.overridden)})
        return transition, next_normalized

    def update(self) -> None:
        if len(self.replay) < self.config.training.batch_size:
            return
        batch = self.replay.sample(self.config.training.batch_size)
        obs = torch.as_tensor(batch.observation["fused"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(
            batch.next_observation["fused"], dtype=torch.float32, device=self.device
        )
        rewards = torch.as_tensor(batch.reward, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.done, dtype=torch.float32, device=self.device)
        policy_out = self.policy(obs)
        with torch.no_grad():
            target_value = rewards + self.config.training.gamma * (1 - dones) * self.policy(next_obs).value
        advantages = target_value - policy_out.value
        policy_loss = -(policy_out.log_prob * advantages.detach()).mean()
        value_loss = F.mse_loss(policy_out.value, target_value.detach())
        entropy_loss = -policy_out.entropy.mean()
        loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.perception.parameters()) + list(self.policy.parameters()),
            self.config.training.gradient_clip,
        )
        self.optimizer.step()
        LOGGER.info(
            "loss=%.4f policy=%.4f value=%.4f entropy=%.4f",
            loss.item(),
            policy_loss.item(),
            value_loss.item(),
            entropy_loss.item(),
        )

    def train(self, steps: Optional[int] = None) -> None:
        total_steps = steps or self.config.training.max_steps
        observation = self.env.reset()
        observation = self.normalize_observations(observation)
        for step in range(total_steps):
            transition, next_obs = self.collect_step(observation)
            self.replay.add(transition)
            observation = next_obs
            if step % self.config.training.log_interval == 0:
                LOGGER.info("Collected step %d", step)
            self.update()
            if step % self.config.training.checkpoint_interval == 0 and step > 0:
                self.save_checkpoint(Path("checkpoints"), step)

    def rollout(self, steps: int) -> None:
        """Run the agent in inference mode without updating weights."""

        observation = self.env.reset()
        observation = self.normalize_observations(observation)
        for step in range(steps):
            normalized = self.normalize_observations(observation)
            perception_output = self.perception(normalized)
            features = perception_output.fused_features.to(self.device)
            policy_output = self.policy.act(features)
            command = self.controller.action_to_control(
                policy_output.action.detach().cpu().numpy()
            )
            speed = getattr(self.env, "get_speed", lambda: 0.0)()
            decision = self.supervisor.enforce(command, perception_output, speed)
            raw_command = getattr(self.controller, "to_carla_control", lambda x: x)(
                decision.command
            )
            observation, _, done, _ = self.env.step(raw_command)
            if done:
                LOGGER.info("Episode finished at step %d", step)
                observation = self.env.reset()
            observation = self.normalize_observations(observation)

    def save_checkpoint(self, directory: Path, step: int) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "policy": self.policy.state_dict(),
            "perception": self.perception.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step,
            "config": self.config,
        }
        path = directory / f"checkpoint_{step}.pt"
        torch.save(checkpoint, path)
        LOGGER.info("Checkpoint saved at %s", path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Universal Driver Trainer")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    args = parser.parse_args()

    env = CarlaEnvAdapter(host=args.host, port=args.port, timeout=args.timeout)
    trainer = Trainer(env, DEFAULT_AGENT_CONFIG, log_dir=args.log_dir)
    trainer.train(steps=args.steps)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

