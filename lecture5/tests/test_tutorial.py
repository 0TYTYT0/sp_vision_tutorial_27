from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
import rerun as rr

import main_rotation
import main_translation
from aim.solver import Solver
from sim.scene import Camera, Robot, limit_rad, wrap_angle
from utils.visualizer import ARMOR_MODEL_PATH, RerunVisualizer


@pytest.fixture
def curves(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, float]]:
    samples: list[dict[str, float]] = []

    class MetricsVisualizer:
        def __init__(self, application_id: str, metric_groups: tuple[str, str]) -> None:
            pass

        def set_time(self, seconds: float) -> None:
            pass

        def log_camera(self, camera: Camera) -> None:
            pass

        def log_robot(self, robot: Robot, *, estimated: bool = False) -> None:
            pass

        def log_observation(
            self, camera: Camera, robot: Robot, image_points: np.ndarray
        ) -> None:
            pass

        def log_scalars(self, values: Mapping[str, float]) -> None:
            samples.append(dict(values))

        def close(self) -> None:
            pass

    monkeypatch.setattr(main_translation, "RerunVisualizer", MetricsVisualizer)
    monkeypatch.setattr(main_rotation, "RerunVisualizer", MetricsVisualizer)
    return samples


def test_model_loads_after_changing_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    model_paths: list[str] = []

    def spawn(self: rr.RecordingStream, **kwargs: object) -> None:
        pass

    def log(
        self: rr.RecordingStream, path: str, data: object, *, static: bool = False
    ) -> None:
        if isinstance(data, rr.Asset3D):
            model_paths.append(path)

    monkeypatch.setattr(rr.RecordingStream, "spawn", spawn)
    monkeypatch.setattr(rr.RecordingStream, "log", log)
    assert ARMOR_MODEL_PATH.parent.name == "assets"
    assert ARMOR_MODEL_PATH.parent.parent.name == "lecture5"
    assert ARMOR_MODEL_PATH.read_bytes()[:4] == b"glTF"
    viewer = RerunVisualizer("test_model_path", ("x", "y"))
    viewer.log_robot(Robot(np.array([3.0, 0.0, 0.1])))
    assert len(model_paths) == 4


@pytest.mark.parametrize("yaw", [0.0, 1.6, 6.2])
def test_camera_solver_recovers_robot_pose(yaw: float) -> None:
    camera = Camera()
    robot = Robot(np.array([3.0, -0.2, 0.1]), yaw)
    armor = robot.observe(camera)
    angle, position = Solver(camera).robot_measurement(
        camera.project(armor), armor.index, robot.radius
    )
    assert abs(wrap_angle(angle - yaw)) < 1e-5
    np.testing.assert_allclose(position, robot.position, atol=1e-5)


def test_translation_keeps_x_fixed_and_tracks_reversals(
    curves: list[dict[str, float]],
) -> None:
    main_translation.run_demo(main_translation.Config(duration=4.0))
    np.testing.assert_array_equal([row["x/truth"] for row in curves], 3.0)
    truth_y = np.array([row["y/truth"] for row in curves])
    assert truth_y.min() >= -0.8
    assert truth_y.max() <= 0.8
    assert np.any(np.diff(truth_y) > 0.0)
    assert np.any(np.diff(truth_y) < 0.0)
    for axis in ("x", "y"):
        errors = np.array(
            [row[f"{axis}/estimate"] - row[f"{axis}/truth"] for row in curves[30:]]
        )
        assert np.sqrt(np.mean(errors**2)) < 0.08


def test_default_rotation_tracks_many_turns(curves: list[dict[str, float]]) -> None:
    config = main_rotation.Config()
    assert config.angular_velocity >= 6.0
    assert config.duration >= 20.0
    main_rotation.run_demo(config)
    truth_angles = np.array([row["angle/truth"] for row in curves])
    assert np.count_nonzero(np.diff(truth_angles) < -np.pi) >= 18
    for name in ("truth", "prediction", "estimate", "measurement"):
        angles = np.array([row[f"angle/{name}"] for row in curves])
        assert np.all((0.0 <= angles) & (angles < 2 * np.pi))
    estimated_angles = np.array([row["angle/estimate"] for row in curves])
    circular_steps = np.array(
        [wrap_angle(float(delta)) for delta in np.diff(estimated_angles)]
    )
    assert np.max(np.abs(circular_steps)) < 0.5
    for axis, tolerance in (("angle", 0.1), ("angular_velocity", 0.15)):
        errors = np.array(
            [row[f"{axis}/estimate"] - row[f"{axis}/truth"] for row in curves[60:]]
        )
        if axis == "angle":
            errors = np.array([wrap_angle(float(error)) for error in errors])
        assert np.sqrt(np.mean(errors**2)) < tolerance


def test_noise_is_reproducible(curves: list[dict[str, float]]) -> None:
    config = main_translation.Config(duration=0.1, seed=7)
    main_translation.run_demo(config)
    first = curves.copy()
    curves.clear()
    main_translation.run_demo(config)
    assert curves == first


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(-0.1, 2 * np.pi - 0.1), (0.0, 0.0), (2 * np.pi, 0.0), (4 * np.pi + 0.2, 0.2)],
)
def test_limit_rad_handles_negative_angles_and_multiple_turns(
    angle: float, expected: float
) -> None:
    assert limit_rad(angle) == pytest.approx(expected)
