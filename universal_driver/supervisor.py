"""Safety supervisor that monitors the driving policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import SupervisorConfig, VehicleConfig
from .control.controller import ControlCommand
from .perception.detector import Detection, PerceptionOutput


@dataclass
class SupervisorDecision:
    command: ControlCommand
    overridden: bool
    reasons: List[str]


class SafetySupervisor:
    """Monitors policy output and enforces safety constraints."""

    def __init__(self, vehicle: VehicleConfig, config: SupervisorConfig) -> None:
        self.vehicle = vehicle
        self.config = config

    def evaluate(self, perception: PerceptionOutput, speed: float) -> List[str]:
        violations: List[str] = []
        for detection in perception.detections:
            distance = np.linalg.norm(np.asarray(detection.position[:2]))
            if distance < self.config.emergency_brake_distance:
                violations.append(
                    f"Emergency brake distance violated by {detection.label} ({distance:.2f} m)"
                )
            ttc = self._time_to_collision(distance, speed, detection)
            if ttc is not None and ttc < self.config.min_follow_time:
                violations.append(
                    f"Time-to-collision below threshold with {detection.label} ({ttc:.2f} s)"
                )
        return violations

    def enforce(
        self,
        command: ControlCommand,
        perception: Optional[PerceptionOutput],
        speed: float,
    ) -> SupervisorDecision:
        if perception is None:
            return SupervisorDecision(command=command, overridden=False, reasons=[])
        reasons = self.evaluate(perception, speed)
        overridden = bool(reasons) and self.config.enable_hard_overrides
        safe_command = command
        if overridden:
            safe_command = ControlCommand(throttle=0.0, brake=self.vehicle.max_brake, steer=0.0)
        return SupervisorDecision(command=safe_command, overridden=overridden, reasons=reasons)

    def _time_to_collision(
        self, distance: float, ego_speed: float, detection: Detection
    ) -> Optional[float]:
        rel_speed = ego_speed - detection.velocity[0]
        if rel_speed <= 1e-3:
            return None
        return max(distance / rel_speed, 0.0)

