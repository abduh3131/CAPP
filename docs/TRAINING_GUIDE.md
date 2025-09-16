# Training Guide

This document describes how to train, evaluate and extend the Universal Driver
inside the CARLA simulator. The workflow is split into five phases.

## 1. Configure Sensors and Vehicle

Edit ``universal_driver/config.py`` or create your own script that instantiates
``AgentConfig`` with the desired sensors, vehicle parameters and network
hyper-parameters. The defaults target a front-camera only vehicle.

## 2. Launch CARLA

Start the CARLA server before training:

```bash
./CarlaUE4.sh -quality-level=Epic
```

Verify the server listens on the host/port that you plan to use.

## 3. Train the Agent

```bash
python -m universal_driver.training.train --steps 50000 --log-dir runs/experiment1
```

Key arguments:

* ``--steps`` controls the number of environment interactions.
* ``--log-dir`` stores training logs and metrics.
* ``--timeout`` adjusts the CARLA RPC timeout for slow machines.

During training checkpoints are written to ``checkpoints/``. You can resume or
fine-tune from the latest checkpoint by editing the script to load the saved
state dictionaries before calling ``train``.

## 4. Evaluate in Closed Loop

```bash
python -m universal_driver.evaluation.run_agent checkpoints/checkpoint_50000.pt --steps 2000
```

The evaluation script restores the saved models, connects to CARLA and performs
inference-only rollouts guarded by the safety supervisor.

## 5. Extending the Stack

* **Additional sensors:** implement a class that inherits from
  ``SensorAdapter`` and register it in ``SENSOR_REGISTRY``.
* **Custom policies:** subclass ``UniversalPolicy`` or provide a new planner
  module; the trainer only requires the ``act`` interface.
* **Safety constraints:** adjust ``SupervisorConfig`` or modify
  ``SafetySupervisor`` to reflect jurisdiction-specific rules.

Logs, checkpoints and sensor recordings can be analysed offline to refine
behaviour, integrate imitation learning or build curriculum training regimes.
