# Universal Driver

The Universal Driver project provides a modular reinforcement learning and
perception stack capable of controlling a wide variety of vehicles inside the
CARLA simulator. It fuses arbitrary sensors (camera, LiDAR, IMU, ultrasonic,
etc.), plans actions with an actor-critic policy and enforces a safety
supervisor that overrides unsafe commands in real time.

## Key Components

* **Configuration:** expressive dataclasses describe sensors, vehicles and
  training hyper-parameters (`universal_driver/config.py`).
* **Sensors:** adapters translate raw simulator feeds to model-ready tensors and
  fall back to safe defaults when data is missing (`universal_driver/sensors`).
* **Perception:** an attention-based fusion model performs detection and builds a
  latent world representation from all sensors
  (`universal_driver/perception/detector.py`).
* **Planning & Control:** the actor-critic policy outputs continuous control
  actions that are translated into throttle/steer/brake commands for any vehicle
  geometry (`universal_driver/planning`, `universal_driver/control`).
* **Safety Supervisor:** monitors time-to-collision and proximity, issuing hard
  overrides when risk is detected (`universal_driver/supervisor.py`).
* **Training/Evaluation:** CARLA integration, replay buffer management and
  rollout utilities (`universal_driver/training`, `universal_driver/evaluation`).

## Getting Started

1. Review the [CARLA setup guide](docs/CARLA_SETUP.md) and install the
   dependencies listed in `requirements.txt`.
2. Launch the CARLA server locally (`./CarlaUE4.sh`).
3. Train an agent:

   ```bash
   python -m universal_driver.training.train --steps 50000 --log-dir runs/demo
   ```

4. Evaluate a saved checkpoint:

   ```bash
   python -m universal_driver.evaluation.run_agent checkpoints/checkpoint_50000.pt --steps 2000
   ```

## Documentation

* [Architecture Overview](docs/ARCHITECTURE.md)
* [Training Guide](docs/TRAINING_GUIDE.md)
* [CARLA Setup](docs/CARLA_SETUP.md)

## Repository Structure

```
universal_driver/
  config.py                 # Core configuration schemas
  sensors/                  # Sensor adapters (CARLA, fallbacks)
  perception/               # Multi-modal perception networks
  planning/                 # Policy networks and action sampling
  control/                  # Actuator conversions
  supervisor.py             # Safety logic and overrides
  training/                 # RL training loop and CARLA adapter
  evaluation/               # Checkpoint loading and rollouts
docs/                       # Guides and manuals
tests/                      # Lightweight unit tests
```

## Notes

* The code auto-detects CUDA and falls back to CPU seamlessly.
* Additional sensors can be integrated by registering new adapters in
  `SENSOR_REGISTRY`.
* The safety supervisor is configurable via `SupervisorConfig` to match local
  driving regulations or company policies.
