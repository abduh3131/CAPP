"""Configuration schemas for the Universal Driver stack.

This module defines dataclasses that capture the runtime configuration of
sensors, vehicles, model architectures and training hyper-parameters.
The configuration objects are intentionally expressive so that the stack
can operate on a wide variety of vehicle platforms and sensor suites
without code changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class SensorConfig:
    """Defines a single sensor channel available to the agent.

    Attributes:
        name: Unique identifier for the sensor.
        modality: Semantic type of the sensor (``camera``, ``lidar``,
            ``radar``, ``imu``, ``ultrasonic`` and so on). The modality is
            used to select appropriate encoders.
        shape: Shape of a single observation frame. The format is channel
            first, i.e. ``(channels, height, width)`` for image-like sensors
            and ``(features,)`` for vector sensors.
        frequency: Expected frequency in Hertz for the sensor stream.
        is_primary: Whether the sensor is expected to be present at all
            times. Optional sensors may be dynamically ignored when data is
            missing.
    """

    name: str
    modality: str
    shape: Sequence[int]
    frequency: float
    is_primary: bool = True
    intrinsic: Optional[Dict[str, float]] = None
    extrinsic: Optional[Dict[str, float]] = None


@dataclass
class VehicleConfig:
    """Captures physical attributes of the controlled vehicle."""

    wheel_base: float
    max_steer: float
    max_throttle: float
    max_brake: float
    mass: float
    length: float
    width: float
    controller_frequency: float = 20.0


@dataclass
class TrainingConfig:
    """Hyper-parameters shared across the training stack."""

    buffer_size: int = 200_000
    batch_size: int = 128
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_steps: int = 1_000_000
    warmup_steps: int = 10_000
    rollout_fragment_length: int = 200
    log_interval: int = 500
    checkpoint_interval: int = 10_000
    gradient_clip: float = 1.0
    device: str = "cuda"


@dataclass
class DetectorConfig:
    """Configuration for the perception detector network."""

    hidden_sizes: Sequence[int] = (256, 128)
    dropout: float = 0.1
    anchor_boxes: int = 32
    grid_size: Tuple[int, int] = (64, 64)
    classes: Sequence[str] = (
        "pedestrian",
        "vehicle",
        "bicycle",
        "obstacle",
        "traffic_light",
    )


@dataclass
class PolicyConfig:
    """Defines the architecture of the actor-critic policy."""

    hidden_sizes: Sequence[int] = (512, 256, 128)
    activation: str = "relu"
    action_dim: int = 3
    stochastic: bool = True
    use_lstm: bool = True


@dataclass
class SupervisorConfig:
    """Configuration for the safety supervisor."""

    emergency_brake_distance: float = 6.0
    min_follow_time: float = 1.2
    lateral_margin: float = 1.0
    violation_penalty: float = 10.0
    enable_hard_overrides: bool = True


@dataclass
class AgentConfig:
    """Aggregates all configuration required to build an agent."""

    sensors: List[SensorConfig]
    vehicle: VehicleConfig
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)

    def sensor_by_name(self, name: str) -> SensorConfig:
        """Return the configuration for a named sensor.

        Raises:
            KeyError: If the sensor is not defined.
        """

        for sensor in self.sensors:
            if sensor.name == name:
                return sensor
        raise KeyError(f"Sensor {name!r} not configured")


DEFAULT_CAMERA_SENSOR = SensorConfig(
    name="front_camera",
    modality="camera",
    shape=(3, 256, 256),
    frequency=30.0,
    intrinsic={"fov": 90.0},
)

DEFAULT_VEHICLE = VehicleConfig(
    wheel_base=2.8,
    max_steer=1.0,
    max_throttle=1.0,
    max_brake=1.0,
    mass=1500.0,
    length=4.7,
    width=1.8,
)


DEFAULT_AGENT_CONFIG = AgentConfig(
    sensors=[DEFAULT_CAMERA_SENSOR],
    vehicle=DEFAULT_VEHICLE,
)

