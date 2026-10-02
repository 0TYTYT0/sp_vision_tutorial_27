"""入口二：纯匀速旋转，比较 angle/角速度真值、预测和估计。"""

from dataclasses import dataclass, field

import numpy as np

from aim.solver import Solver
from sim.scene import Camera, Robot, limit_rad, wrap_angle
from utils.ekf import ExtendedKalmanFilter, FloatArray
from utils.visualizer import RerunVisualizer


@dataclass
class Config:
    # 状态排列为 [angle, angular_velocity]，单位 rad、rad/s。
    duration: float = 20.0
    dt: float = 1.0 / 60.0
    seed: int = 42
    position: tuple[float, float, float] = (3.0, 0.0, 0.1)
    initial_angle: float = 0.0
    angular_velocity: float = 6.0
    radius: float = 0.26
    x0: FloatArray = field(default_factory=lambda: np.array([0.4, 0.0]))
    p0: FloatArray = field(default_factory=lambda: np.diag([0.5**2, 2.0**2]))
    pixel_noise_std: float = 1.0
    # R 只有 angle 的方差；角速度没有直接量测，由连续帧中的角度变化估计。
    r: FloatArray = field(default_factory=lambda: np.array([[0.08**2]]))
    process_angular_acceleration_std: float = 0.8


def angle_residual(observed: FloatArray, predicted: FloatArray) -> FloatArray:
    return np.array([wrap_angle(float(observed[0] - predicted[0]))])


def run_demo(config: Config) -> None:
    if config.duration <= 0 or config.dt <= 0 or config.radius <= 0:
        raise ValueError("duration、dt 和 radius 必须为正")
    if config.pixel_noise_std < 0 or config.process_angular_acceleration_std < 0:
        raise ValueError("噪声标准差必须非负")
    if config.x0.shape != (2,) or config.p0.shape != (2, 2) or config.r.shape != (1, 1):
        raise ValueError("旋转场景要求 x0=(2,)、P0=(2,2)、R=(1,1)")
    rng = np.random.default_rng(config.seed)
    camera = Camera()
    solver = Solver(camera)
    ekf = ExtendedKalmanFilter(config.x0, config.p0)
    visualizer = RerunVisualizer(
        "lecture5_rotation",
        ("angle", "angular_velocity"),
    )
    visualizer.log_camera(camera)

    dt = config.dt
    transition = np.array([[1.0, dt], [0.0, 1.0]])
    observation_matrix = np.array([[1.0, 0.0]])
    noise_mapping = np.array([[0.5 * dt**2], [dt]])
    q = noise_mapping @ noise_mapping.T * config.process_angular_acceleration_std**2

    def f(state: FloatArray) -> FloatArray:
        return transition @ state

    def jacobian_f(state: FloatArray) -> FloatArray:
        return transition

    def h(state: FloatArray) -> FloatArray:
        return observation_matrix @ state

    def jacobian_h(state: FloatArray) -> FloatArray:
        return observation_matrix

    timestamps = np.arange(0.0, config.duration, dt)
    for index, timestamp in enumerate(timestamps):
        angle = config.initial_angle + config.angular_velocity * timestamp
        robot = Robot(np.array(config.position), float(angle), config.radius)
        armor = robot.observe(camera)
        points = camera.project(armor) + rng.normal(0.0, config.pixel_noise_std, (4, 2))
        measured_angle, _ = solver.robot_measurement(points, armor.index, robot.radius)
        z = np.array([measured_angle])
        if index > 0:
            ekf.predict(f, jacobian_f, q)
        prediction = ekf.x.copy()
        ekf.update(z, h, jacobian_h, config.r, residual=angle_residual)
        # 滤波内部保留连续角度；展示时限制到 [0, 2π)，每圈回到零。
        visualizer.set_time(float(timestamp))
        visualizer.log_scalars(
            {
                "angle/truth": limit_rad(float(angle)),
                "angle/prediction": limit_rad(float(prediction[0])),
                "angle/estimate": limit_rad(float(ekf.x[0])),
                "angle/measurement": limit_rad(measured_angle),
                "angular_velocity/truth": config.angular_velocity,
                "angular_velocity/prediction": float(prediction[1]),
                "angular_velocity/estimate": float(ekf.x[1]),
            }
        )
        visualizer.log_robot(robot)
        visualizer.log_robot(
            Robot(robot.position, float(ekf.x[0]), robot.radius), estimated=True
        )
        visualizer.log_observation(camera, robot, points)
    visualizer.close()


def main() -> None:
    run_demo(Config())


if __name__ == "__main__":
    main()
