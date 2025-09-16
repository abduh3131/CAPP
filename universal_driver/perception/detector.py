"""Perception stack for the Universal Driver."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from ..config import DetectorConfig, SensorConfig
from ..sensors.base import Observation


@dataclass
class Detection:
    """Represents a detected object in the scene."""

    label: str
    score: float
    position: Tuple[float, float, float]
    size: Tuple[float, float, float]
    velocity: Tuple[float, float, float]


@dataclass
class PerceptionOutput:
    """Aggregated perception features provided to the planner."""

    fused_features: torch.Tensor
    detections: List[Detection]


class SensorEncoder(nn.Module):
    """Encodes sensor observations into latent embeddings."""

    def __init__(self, sensor: SensorConfig, hidden_size: int) -> None:
        super().__init__()
        input_dim = int(torch.tensor(sensor.shape).prod().item())
        self.linear = nn.Linear(input_dim, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, observation: Observation) -> torch.Tensor:
        device = self.linear.weight.device
        x = torch.as_tensor(observation.as_numpy(), dtype=torch.float32)
        x = x.to(device)
        x = x.reshape(1, -1)
        x = self.linear(x)
        return self.norm(torch.tanh(x))


class FusionTransformer(nn.Module):
    """Attention based feature fusion for multiple sensor embeddings."""

    def __init__(self, hidden_size: int, heads: int = 4, layers: int = 2) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.transformer(embeddings)


class DetectionHead(nn.Module):
    """Predicts object bounding boxes on a BEV grid."""

    def __init__(self, config: DetectorConfig, hidden_size: int) -> None:
        super().__init__()
        grid_h, grid_w = config.grid_size
        output_dim = grid_h * grid_w * (len(config.classes) + 7)
        layers: List[nn.Module] = []
        last_dim = hidden_size
        for hidden in config.hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            last_dim = hidden
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)
        self.grid_size = config.grid_size
        self.classes = list(config.classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)

    def decode(self, logits: torch.Tensor, threshold: float = 0.5) -> List[Detection]:
        detections: List[Detection] = []
        grid_h, grid_w = self.grid_size
        logits = logits.view(-1, grid_h * grid_w, len(self.classes) + 7)
        for cell in logits[0]:
            class_scores = torch.sigmoid(cell[: len(self.classes)])
            if class_scores.max().item() < threshold:
                continue
            best_idx = int(torch.argmax(class_scores).item())
            score = float(class_scores[best_idx].item())
            x, y, z, dx, dy, dz, speed = cell[len(self.classes) :].tolist()
            detections.append(
                Detection(
                    label=self.classes[best_idx],
                    score=score,
                    position=(x, y, z),
                    size=(dx, dy, dz),
                    velocity=(speed, 0.0, 0.0),
                )
            )
        return detections


class MultiModalPerception(nn.Module):
    """End-to-end perception module that fuses arbitrary sensors."""

    def __init__(self, sensors: List[SensorConfig], config: DetectorConfig) -> None:
        super().__init__()
        hidden_size = 256
        self.encoders = nn.ModuleDict(
            {sensor.name: SensorEncoder(sensor, hidden_size) for sensor in sensors}
        )
        self.fusion = FusionTransformer(hidden_size)
        self.detection_head = DetectionHead(config, hidden_size)

    def forward(
        self, observations: Dict[str, Observation], threshold: float = 0.5
    ) -> PerceptionOutput:
        embeddings: List[torch.Tensor] = []
        for name, encoder in self.encoders.items():
            if name not in observations:
                continue
            embeddings.append(encoder(observations[name]))
        if not embeddings:
            raise ValueError("No observations available for perception")
        stacked = torch.stack(embeddings, dim=1)
        fused = self.fusion(stacked)
        pooled = fused.mean(dim=1)
        logits = self.detection_head(pooled)
        detections = self.detection_head.decode(logits, threshold=threshold)
        return PerceptionOutput(fused_features=pooled, detections=detections)

