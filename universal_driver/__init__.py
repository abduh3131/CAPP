"""Universal Driver package."""

from .config import (
    AgentConfig,
    DEFAULT_AGENT_CONFIG,
    DEFAULT_CAMERA_SENSOR,
    DEFAULT_VEHICLE,
    DetectorConfig,
    PolicyConfig,
    SensorConfig,
    SupervisorConfig,
    TrainingConfig,
    VehicleConfig,
)

try:  # pragma: no cover - optional import
    from .training.train import Trainer, CarlaEnvAdapter
except Exception:  # pragma: no cover - optional import
    Trainer = None  # type: ignore
    CarlaEnvAdapter = None  # type: ignore

__all__ = [
    "AgentConfig",
    "DEFAULT_AGENT_CONFIG",
    "DEFAULT_CAMERA_SENSOR",
    "DEFAULT_VEHICLE",
    "DetectorConfig",
    "PolicyConfig",
    "SensorConfig",
    "SupervisorConfig",
    "TrainingConfig",
    "VehicleConfig",
]

if Trainer is not None:
    __all__.extend(["Trainer", "CarlaEnvAdapter"])
