"""Sensor abstractions for the Universal Driver stack."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..config import SensorConfig


@dataclass
class Observation:
    """Represents a single observation from a sensor."""

    sensor: SensorConfig
    data: Any
    timestamp: float

    def as_numpy(self) -> np.ndarray:
        """Convert the observation payload into a NumPy array."""

        if isinstance(self.data, np.ndarray):
            return self.data
        if hasattr(self.data, "numpy"):
            return self.data.numpy()  # type: ignore[no-any-return]
        return np.asarray(self.data)


class SensorAdapter(abc.ABC):
    """Base class for all sensor adapters.

    An adapter is responsible for subscribing to a data source (such as a
    CARLA sensor, ROS topic or CAN frame) and exposing sensor data in a
    unified format to the rest of the stack.
    """

    def __init__(self, config: SensorConfig):
        self.config = config

    @abc.abstractmethod
    def initialize(self, world: Optional[Any] = None, parent: Optional[Any] = None) -> None:
        """Perform any setup necessary prior to data collection."""

    @abc.abstractmethod
    def read(self) -> Observation:
        """Retrieve the latest observation from the sensor."""

    def to_feature_vector(self, observation: Observation) -> np.ndarray:
        """Convert raw sensor data into a feature vector suitable for models."""

        return observation.as_numpy().reshape(-1)

    def destroy(self) -> None:
        """Clean up any resources held by the adapter."""

        return None


class NullSensorAdapter(SensorAdapter):
    """A fallback adapter that returns zeroed-out data.

    This is primarily useful for testing and when certain sensors are
    unavailable at runtime. The adapter respects the configured observation
    shape and produces zero tensors at the expected frequency.
    """

    def __init__(self, config: SensorConfig):
        super().__init__(config)
        self._last_timestamp: float = 0.0

    def initialize(self, world: Optional[Any] = None, parent: Optional[Any] = None) -> None:
        self._last_timestamp = 0.0

    def read(self) -> Observation:
        shape = tuple(self.config.shape)
        data = np.zeros(shape, dtype=np.float32)
        self._last_timestamp += 1.0 / max(self.config.frequency, 1.0)
        return Observation(sensor=self.config, data=data, timestamp=self._last_timestamp)


class SensorRegistry:
    """Registry of sensor adapters available to the agent."""

    def __init__(self) -> None:
        self._factories: Dict[str, type[SensorAdapter]] = {}

    def register(self, modality: str, adapter_cls: type[SensorAdapter]) -> None:
        self._factories[modality] = adapter_cls

    def create(self, config: SensorConfig) -> SensorAdapter:
        adapter_cls = self._factories.get(config.modality, NullSensorAdapter)
        return adapter_cls(config)


SENSOR_REGISTRY = SensorRegistry()
