"""Low level controllers for different vehicle types."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import VehicleConfig


@dataclass
class ControlCommand:
    throttle: float
    brake: float
    steer: float


class UniversalController:
    """Generic controller that understands normalized policy actions."""

    def __init__(self, vehicle: VehicleConfig) -> None:
        self.vehicle = vehicle

    def action_to_control(self, action: np.ndarray) -> ControlCommand:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        steer, throttle, brake = action.tolist()[:3]
        steer_cmd = float(np.clip(steer, -1.0, 1.0) * self.vehicle.max_steer)
        throttle_cmd = float(np.clip(throttle, 0.0, 1.0) * self.vehicle.max_throttle)
        brake_cmd = float(np.clip(brake, 0.0, 1.0) * self.vehicle.max_brake)
        return ControlCommand(throttle=throttle_cmd, brake=brake_cmd, steer=steer_cmd)

    def stabilize(self, command: ControlCommand, speed: float) -> ControlCommand:
        """Apply simple heuristics to reduce oscillations at low speed."""

        if speed < 2.0:
            command.steer *= 0.5
        return command

    def to_carla_control(self, command: ControlCommand) -> Optional["carla.VehicleControl"]:
        try:  # pragma: no cover - optional dependency
            import carla  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            return None
        control = carla.VehicleControl()
        control.throttle = float(np.clip(command.throttle, 0.0, 1.0))
        control.brake = float(np.clip(command.brake, 0.0, 1.0))
        control.steer = float(np.clip(command.steer / max(self.vehicle.max_steer, 1e-3), -1.0, 1.0))
        return control

