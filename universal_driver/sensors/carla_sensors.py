"""CARLA specific sensor adapters."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import carla  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    carla = None  # type: ignore

from ..config import SensorConfig
from .base import Observation, SensorAdapter, SENSOR_REGISTRY

LOGGER = logging.getLogger(__name__)


class CarlaSensorAdapter(SensorAdapter):
    """Base class for adapters backed by CARLA sensors."""

    def __init__(self, config: SensorConfig):
        super().__init__(config)
        self._world: Optional[Any] = None
        self._sensor: Optional[Any] = None
        self._parent: Optional[Any] = None
        self._latest_measurement: Optional[Dict[str, Any]] = None

    def initialize(self, world: Optional[Any] = None, parent: Optional[Any] = None) -> None:
        if carla is None:
            raise RuntimeError("carla package is not available")
        if world is None:
            raise ValueError("CARLA world handle is required")
        self._world = world
        self._parent = parent
        self._spawn_sensor()

    def _spawn_sensor(self) -> None:
        assert carla is not None
        assert self._world is not None
        blueprint_library = self._world.get_blueprint_library()
        blueprint = blueprint_library.find(self._blueprint_id)
        for key, value in (self.config.intrinsic or {}).items():
            blueprint.set_attribute(key, str(value))
        transform = carla.Transform()
        extrinsic = self.config.extrinsic or {}
        transform.location = carla.Location(
            x=extrinsic.get("x", 0.0),
            y=extrinsic.get("y", 0.0),
            z=extrinsic.get("z", 0.0),
        )
        transform.rotation = carla.Rotation(
            pitch=extrinsic.get("pitch", 0.0),
            yaw=extrinsic.get("yaw", 0.0),
            roll=extrinsic.get("roll", 0.0),
        )
        spawn_kwargs = {}
        if self._parent is not None:
            spawn_kwargs["attach_to"] = self._parent
        self._sensor = self._world.spawn_actor(blueprint, transform, **spawn_kwargs)
        self._sensor.listen(self._on_sensor_event)

    def _on_sensor_event(self, event: Any) -> None:
        self._latest_measurement = {
            "raw": event,
            "timestamp": float(getattr(event, "timestamp", 0.0)),
        }

    def destroy(self) -> None:
        if self._sensor is not None:
            try:
                self._sensor.stop()
            except Exception:
                LOGGER.debug("Failed to stop sensor %s", self.config.name)
            self._sensor.destroy()
            self._sensor = None
            self._latest_measurement = None

    def read(self) -> Observation:
        if self._latest_measurement is None:
            zeros = np.zeros(tuple(self.config.shape), dtype=np.float32)
            return Observation(sensor=self.config, data=zeros, timestamp=0.0)
        data = self._to_numpy(self._latest_measurement["raw"])
        return Observation(
            sensor=self.config,
            data=data,
            timestamp=self._latest_measurement.get("timestamp", 0.0),
        )

    def _to_numpy(self, raw: Any) -> np.ndarray:
        raise NotImplementedError


class CarlaCameraAdapter(CarlaSensorAdapter):
    _blueprint_id = "sensor.camera.rgb"

    def _to_numpy(self, raw: Any) -> np.ndarray:
        image = np.frombuffer(raw.raw_data, dtype=np.uint8)
        image = image.reshape((raw.height, raw.width, 4))[:, :, :3]
        image = np.transpose(image, (2, 0, 1))
        return image.astype(np.float32) / 255.0


class CarlaLidarAdapter(CarlaSensorAdapter):
    _blueprint_id = "sensor.lidar.ray_cast"

    def _to_numpy(self, raw: Any) -> np.ndarray:
        points = np.frombuffer(raw.raw_data, dtype=np.float32)
        return points.reshape(-1, 4)


class CarlaIMUAdapter(CarlaSensorAdapter):
    _blueprint_id = "sensor.other.imu"

    def _to_numpy(self, raw: Any) -> np.ndarray:
        return np.asarray(
            [
                raw.accelerometer.x,
                raw.accelerometer.y,
                raw.accelerometer.z,
                raw.gyroscope.x,
                raw.gyroscope.y,
                raw.gyroscope.z,
            ],
            dtype=np.float32,
        )


SENSOR_REGISTRY.register("camera", CarlaCameraAdapter)
SENSOR_REGISTRY.register("lidar", CarlaLidarAdapter)
SENSOR_REGISTRY.register("imu", CarlaIMUAdapter)
