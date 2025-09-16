# Universal Driver Architecture

The Universal Driver stack is organised into three conceptual layers that can be
combined to operate vehicles with arbitrary sensor suites.

## 1. Sensor Abstraction

* **Adapters** translate raw simulator or real-world sensor outputs to a unified
  representation (`Observation`).
* **Registry** resolves sensor modalities to concrete adapters. New hardware can
  be integrated by registering a class that implements the `SensorAdapter`
  interface.

## 2. Intelligence Core

* **Perception (`universal_driver/perception`)** performs multi-modal feature
  extraction, attention-based fusion and dense object detection. The output is a
  latent tensor and a set of structured detections that describe nearby
  obstacles, actors and traffic agents.
* **Planning (`universal_driver/planning`)** provides an actor-critic policy
  capable of stochastic or deterministic control. It optionally supports LSTM
  recurrence, enabling the agent to reason over time.
* **Control (`universal_driver/control`)** converts abstract actions to concrete
  actuator commands for throttle, brake and steering across different vehicle
  geometries.
* **Supervision (`universal_driver/supervisor`)** wraps the policy with a safety
  layer. It monitors planned actions, enforces minimum distance and
  time-to-collision constraints, and can hard-override controls when risk is
  detected.

## 3. Training & Runtime

* **Training (`universal_driver/training`)** orchestrates the CARLA simulation,
  sensor data collection, replay buffer management and policy optimisation.
  Training uses off-policy actor-critic updates with entropy regularisation.
* **Evaluation (`universal_driver/evaluation`)** loads checkpoints and runs the
  trained agent in closed-loop mode, delegating safety to the supervisor.

The configuration system (`universal_driver/config.py`) drives the entire stack,
allowing the same codebase to support scooters, karts or heavy vehicles simply
by supplying the relevant physical and sensor parameters.
