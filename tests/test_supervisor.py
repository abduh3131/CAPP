import pytest

torch = pytest.importorskip("torch")

from universal_driver.config import SupervisorConfig, VehicleConfig
from universal_driver.control.controller import ControlCommand
from universal_driver.perception.detector import Detection, PerceptionOutput
from universal_driver.supervisor import SafetySupervisor


def make_perception(distance: float) -> PerceptionOutput:
    detection = Detection(
        label="pedestrian",
        score=0.9,
        position=(distance, 0.0, 0.0),
        size=(0.5, 0.5, 1.7),
        velocity=(0.0, 0.0, 0.0),
    )
    fused = torch.ones((1, 256), dtype=torch.float32)
    return PerceptionOutput(fused_features=fused, detections=[detection])


def test_supervisor_triggers_brake():
    vehicle = VehicleConfig(
        wheel_base=2.5,
        max_steer=1.0,
        max_throttle=1.0,
        max_brake=1.0,
        mass=1200.0,
        length=4.0,
        width=1.8,
    )
    config = SupervisorConfig(emergency_brake_distance=5.0)
    supervisor = SafetySupervisor(vehicle, config)
    command = ControlCommand(throttle=0.5, brake=0.0, steer=0.0)
    perception = make_perception(3.0)
    decision = supervisor.enforce(command, perception, speed=10.0)
    assert decision.overridden is True
    assert decision.command.brake == vehicle.max_brake

