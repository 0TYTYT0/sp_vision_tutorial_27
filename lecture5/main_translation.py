"""入口一：固定 x，沿 y 轴匀速往返，比较 x/y 真值、预测与估计。"""

from dataclasses import dataclass, field

import numpy as np

from aim.solver import Solver
from sim.scene import Camera, Robot
from utils.ekf import ExtendedKalmanFilter, FloatArray
from utils.visualizer import RerunVisualizer


@dataclass
class Config:
    # 单位：米、秒、弧度。状态排列为 [x, vx, y, vy]。
    duration: float = 10.0
    dt: float = 1.0 / 60.0
    seed: int = 42
    initial_position: tuple[float, float, float] = (3.0, -0.8, 0.1)
    y_limits: tuple[float, float] = (-0.8, 0.8)
    translation_speed: float = 1.2
    robot_yaw: float = 0.0
    radius: float = 0.26
    # x0 是滤波器的猜测；P0 对角线为方差，速度也可设置为未知。
    x0: FloatArray = field(default_factory=lambda: np.array([2.5, 0.0, -0.2, 0.0]))
    p0: FloatArray = field(
        default_factory=lambda: np.diag([0.5**2, 1.0**2, 0.5**2, 1.0**2])
    )
    # 实际角点噪声：像素标准差；R 是解算后 x/y 量测的假设方差，两者不能直接等同。
    pixel_noise_std: float = 1.0
    r: FloatArray = field(default_factory=lambda: np.diag([0.03**2, 0.01**2]))
    # Q = G diag(加速度标准差²) Gᵀ；换向时模型失配，较大的 y 轴 Q 加快响应。
    process_acceleration_std: tuple[float, float] = (0.5, 3.0)


def translation_truth(config: Config, timestamp: float) -> FloatArray:
    """三角波位置，端点立即反向；即使 dt 跨过端点也不越界。"""
    lower, upper = config.y_limits
    span = upper - lower
    phase = (
        config.initial_position[1] - lower + config.translation_speed * timestamp
    ) % (2 * span)
    if phase < span:
        y, vy = lower + phase, config.translation_speed
    else:
        y, vy = upper - (phase - span), -config.translation_speed
    return np.array([config.initial_position[0], 0.0, y, vy])


def run_demo(config: Config) -> None:
    if config.duration <= 0 or config.dt <= 0 or config.radius <= 0:
        raise ValueError("duration、dt 和 radius 必须为正")
    lower, upper = config.y_limits
    if lower >= upper or not lower <= config.initial_position[1] <= upper:
        raise ValueError("y_limits 必须递增，初始 y 必须在往返区间内")
    if config.translation_speed <= 0:
        raise ValueError("translation_speed 必须为正")
    if config.pixel_noise_std < 0 or min(config.process_acceleration_std) < 0:
        raise ValueError("噪声标准差必须非负")
    if config.x0.shape != (4,) or config.p0.shape != (4, 4) or config.r.shape != (2, 2):
        raise ValueError("平移场景要求 x0=(4,)、P0=(4,4)、R=(2,2)")
    rng = np.random.default_rng(config.seed)
    camera = Camera()
    solver = Solver(camera)
    ekf = ExtendedKalmanFilter(config.x0, config.p0)
    visualizer = RerunVisualizer(
        "lecture5_translation",
        ("x", "y"),
    )
    visualizer.log_camera(camera)

    dt = config.dt
    # f(x) = F x：匀速模型。h(x) = H x：位姿解算后只观测位置。
    transition = np.array(
        [
            [1.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, dt],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    observation_matrix = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    noise_mapping = np.array(
        [[0.5 * dt**2, 0.0], [dt, 0.0], [0.0, 0.5 * dt**2], [0.0, dt]]
    )
    q = (
        noise_mapping
        @ np.diag(np.square(config.process_acceleration_std))
        @ noise_mapping.T
    )

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
        truth = translation_truth(config, float(timestamp))
        robot = Robot(
            np.array([truth[0], truth[2], config.initial_position[2]]),
            config.robot_yaw,
            config.radius,
        )
        armor = robot.observe(camera)
        points = camera.project(armor) + rng.normal(0.0, config.pixel_noise_std, (4, 2))
        _, measured_position = solver.robot_measurement(
            points, armor.index, robot.radius
        )
        z = measured_position[:2]
        if index > 0:
            ekf.predict(f, jacobian_f, q)
        prediction = ekf.x.copy()  # x⁻：看到本帧量测之前的预测。
        ekf.update(z, h, jacobian_h, config.r)
        visualizer.set_time(float(timestamp))
        visualizer.log_scalars(
            {
                "x/truth": float(truth[0]),
                "x/prediction": float(prediction[0]),
                "x/estimate": float(ekf.x[0]),
                "x/measurement": float(z[0]),
                "y/truth": float(truth[2]),
                "y/prediction": float(prediction[2]),
                "y/estimate": float(ekf.x[2]),
                "y/measurement": float(z[1]),
            }
        )
        visualizer.log_robot(robot)
        visualizer.log_robot(
            Robot(
                np.array([ekf.x[0], ekf.x[2], robot.position[2]]),
                robot.yaw,
                robot.radius,
            ),
            estimated=True,
        )
        visualizer.log_observation(camera, robot, points)
    visualizer.close()


def main() -> None:
    run_demo(Config())


if __name__ == "__main__":
    main()
