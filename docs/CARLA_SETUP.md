# CARLA Simulator Setup

The Universal Driver was designed to operate in the [CARLA](https://carla.org/)
simulator first. The following steps configure a development environment.

## 1. Install CARLA

1. Download the CARLA binary package that matches your platform.
2. Unpack it and add the extracted directory to your ``PATH`` or launch CARLA
   manually before training the agent: ``./CarlaUE4.sh -quality-level=Epic``.

Alternatively you may build from source following the official CARLA
instructions. Ensure the simulator is running on ``127.0.0.1:2000`` (default).

## 2. Python Dependencies

Create a virtual environment and install the project requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The requirements include the CARLA Python API wheel. If you compiled CARLA from
source you might need to install the wheel located in ``PythonAPI/carla/dist``
manually.

## 3. Running Sensors

The trainer automatically registers the sensors defined in
``config.AgentConfig``. Ensure the blueprint IDs exist in your CARLA build. You
can add additional sensors by editing the configuration or providing custom
adapters.

## 4. Troubleshooting

* **Timeouts:** increase the ``--timeout`` argument when launching the trainer.
* **Missing wheel:** verify that ``pip show carla`` returns a package and matches
  the CARLA server version.
* **GPU usage:** set ``training.device`` to ``cpu`` in the configuration if a GPU
  is unavailable.
