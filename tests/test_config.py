from universal_driver.config import AgentConfig, DEFAULT_AGENT_CONFIG, DEFAULT_VEHICLE, SensorConfig


def test_sensor_lookup():
    config = DEFAULT_AGENT_CONFIG
    sensor = config.sensor_by_name("front_camera")
    assert isinstance(sensor, SensorConfig)
    assert sensor.name == "front_camera"


def test_additional_sensor_lookup():
    sensors = [
        SensorConfig(name="front_camera", modality="camera", shape=(3, 256, 256), frequency=30.0),
        SensorConfig(name="imu", modality="imu", shape=(6,), frequency=100.0),
    ]
    config = AgentConfig(sensors=sensors, vehicle=DEFAULT_VEHICLE)
    sensor = config.sensor_by_name("imu")
    assert sensor.modality == "imu"

