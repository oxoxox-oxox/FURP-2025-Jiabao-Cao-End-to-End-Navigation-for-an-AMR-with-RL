#!/usr/bin/env python

"""
NeupanCore — 官方 neupan_ros2 节点 + 延迟补偿（最小改动版）

在官方节点基础上新增（全部用 # === [DELAY-COMP] === 标记）:
  1. 未来状态估计: 传入neupan前, 用 dt = 1/control_frequency 预测单步后的状态
     (速度来源: /odom 实测速度优先, 回退到上一条下发指令)
  2. 指令下发频率 = 控制频率: 若推理耗时 < 控制周期, sleep到周期边界再发布,
     使指令恰好在机器人到达预测位姿的时刻生效
     (sleep上限=1个控制周期(50ms@20Hz), 控制timer在独立MutuallyExclusive组,
      不阻塞scan/odom/goal回调 —— 区别于旧版sleep整个prediction_horizon的实现)

yaml新增参数:
  use_delay_compensation: true    # 补偿开关(A/B对比)
  odom_topic: '/odom'
"""
import os
import threading
import time
import traceback
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import numpy.typing as npt
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from ament_index_python.packages import get_package_share_directory
import tf2_ros

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan

try:
    from neupan import neupan
    from neupan.util import get_transform
except ImportError as e:
    raise ImportError(
        f"Failed to import 'neupan' package: {e}. Please install NeuPAN first."
    ) from e

from neupan_ros2.visualization_manager import VisualizationManager
from neupan_ros2.utils import yaw_to_quat, quat_to_yaw


class NeupanCore(Node):
    """ROS2 node for NeuPAN navigation algorithm with delay compensation."""

    def __init__(self) -> None:
        super().__init__("neupan_node")

        self._state_lock = threading.Lock()
        self.control_group = MutuallyExclusiveCallbackGroup()
        self.callback_group = ReentrantCallbackGroup()
        self.pkg_dir = get_package_share_directory("neupan_ros2")

        # ---- 官方参数 ----
        self.declare_parameter("robot_type", "")
        self.declare_parameter("robot_description", "")
        self.declare_parameter("robot_config_dir", "")
        self.declare_parameter("planner_config_file", "planner.yaml")
        self.declare_parameter("dune_checkpoint_file", "models/dune_model_5000.pth")
        self.declare_parameter("neupan_config_file", "NOT SET")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("lidar_frame", "laser_link")
        self.declare_parameter("marker_size", 0.05)
        self.declare_parameter("marker_z", 1.0)
        self.declare_parameter("scan_angle_max", 3.14)
        self.declare_parameter("scan_angle_min", -3.14)
        self.declare_parameter("scan_downsample", 1)
        self.declare_parameter("scan_range_min", 0.1)
        self.declare_parameter("scan_range_max", 5.0)
        self.declare_parameter("refresh_initial_path", False)
        self.declare_parameter("flip_angle", False)
        self.declare_parameter("include_initial_path_direction", False)
        self.declare_parameter("control_frequency", 50.0)

        # === [DELAY-COMP] 新增参数 ===
        self.declare_parameter("use_delay_compensation", True)
        self.declare_parameter("odom_topic", "/odom")

        self.declare_parameter("enable_visualization", True)
        self.declare_parameter("enable_dune_markers", True)
        self.declare_parameter("enable_nrmp_markers", True)
        self.declare_parameter("enable_robot_marker", True)
        self.declare_parameter("cmd_vel_topic", "/neupan_cmd_vel")
        self.declare_parameter("plan_output_topic", "/neupan_plan")
        self.declare_parameter("ref_state_topic", "/neupan_ref_state")
        self.declare_parameter("initial_path_topic", "/neupan_initial_path")
        self.declare_parameter("dune_markers_topic", "/dune_point_markers")
        self.declare_parameter("robot_marker_topic", "/robot_marker")
        self.declare_parameter("nrmp_markers_topic", "/nrmp_point_markers")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("plan_input_topic", "/plan")
        self.declare_parameter("goal_topic", "/goal_pose")

        # ---- 配置加载（官方逻辑不变） ----
        robot_config_dir = self.get_parameter("robot_config_dir").get_parameter_value().string_value
        if not robot_config_dir or not os.path.isdir(robot_config_dir):
            raise ValueError(
                f"Invalid robot_config_dir: '{robot_config_dir}'. Must be set by launch file."
            )
        robot_type = self.get_parameter("robot_type").get_parameter_value().string_value
        self.get_logger().info(f"Loading robot configuration: {robot_type}")

        planner_config_file = self.get_parameter("planner_config_file").get_parameter_value().string_value
        self.planner_config_file = os.path.join(robot_config_dir, planner_config_file)
        dune_checkpoint_file = self.get_parameter("dune_checkpoint_file").get_parameter_value().string_value
        self.dune_checkpoint = os.path.join(robot_config_dir, dune_checkpoint_file)

        if not os.path.isfile(self.planner_config_file):
            raise FileNotFoundError(f"Planner config not found: {self.planner_config_file}")
        if not os.path.isfile(self.dune_checkpoint):
            raise FileNotFoundError(f"DUNE checkpoint not found: {self.dune_checkpoint}")

        self.map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.lidar_frame = self.get_parameter("lidar_frame").get_parameter_value().string_value
        self.marker_size = self.get_parameter("marker_size").get_parameter_value().double_value
        self.marker_z = self.get_parameter("marker_z").get_parameter_value().double_value
        self.scan_range = np.array([
            self.get_parameter("scan_range_min").get_parameter_value().double_value,
            self.get_parameter("scan_range_max").get_parameter_value().double_value])
        self.scan_angle_range = np.array([
            self.get_parameter("scan_angle_min").get_parameter_value().double_value,
            self.get_parameter("scan_angle_max").get_parameter_value().double_value])
        self.scan_downsample = self.get_parameter("scan_downsample").get_parameter_value().integer_value
        self.refresh_initial_path = self.get_parameter("refresh_initial_path").get_parameter_value().bool_value
        self.flip_angle = self.get_parameter("flip_angle").get_parameter_value().bool_value
        self.include_initial_path_direction = self.get_parameter(
            "include_initial_path_direction").get_parameter_value().bool_value
        self.enable_visualization = self.get_parameter("enable_visualization").get_parameter_value().bool_value
        self.enable_dune_markers = self.get_parameter("enable_dune_markers").get_parameter_value().bool_value
        self.enable_nrmp_markers = self.get_parameter("enable_nrmp_markers").get_parameter_value().bool_value
        self.enable_robot_marker = self.get_parameter("enable_robot_marker").get_parameter_value().bool_value

        pan = {'dune_checkpoint': self.dune_checkpoint}
        self.neupan_planner = neupan.init_from_yaml(self.planner_config_file, pan=pan)
        self.get_logger().info(
            f"Robot: {self.neupan_planner.robot.length:.3f}m x {self.neupan_planner.robot.width:.3f}m, "
            f"kinematics: {self.neupan_planner.robot.kinematics}")
        self.get_logger().info("NeuPAN planner initialized successfully")

        # ---- 共享状态 ----
        self.obstacle_points: Optional[npt.NDArray] = None
        self.robot_state: Optional[npt.NDArray] = None
        self.stop: bool = False
        self.arrive: bool = False
        self.goal: Optional[npt.NDArray] = None

        # === [DELAY-COMP] 补偿状态 ===
        self.current_velocity: npt.NDArray = np.zeros((2, 1))   # /odom实测 [v, w]
        self.last_commanded_vel: npt.NDArray = np.zeros((2, 1)) # 上一条下发指令
        self.last_odom_time: float = 0.0
        self._overrun_count: int = 0

        # ---- 发布器 ----
        gp = lambda n: self.get_parameter(n).get_parameter_value().string_value
        self.vel_pub = self.create_publisher(Twist, gp("cmd_vel_topic"), 10)
        self.plan_pub = self.create_publisher(Path, gp("plan_output_topic"), 10)
        self.ref_state_pub = self.create_publisher(Path, gp("ref_state_topic"), 10)
        self.ref_path_pub = self.create_publisher(Path, gp("initial_path_topic"), 10)

        viz_config = {
            'enable_visualization': self.enable_visualization,
            'enable_dune_markers': self.enable_dune_markers,
            'enable_nrmp_markers': self.enable_nrmp_markers,
            'enable_robot_marker': self.enable_robot_marker,
            'map_frame': self.map_frame,
            'marker_size': self.marker_size,
            'marker_z': self.marker_z,
            'dune_markers_topic': gp("dune_markers_topic"),
            'nrmp_markers_topic': gp("nrmp_markers_topic"),
            'robot_marker_topic': gp("robot_marker_topic"),
            'state_lock': self._state_lock,
        }
        self.viz_manager = VisualizationManager(self, viz_config)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        scan_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(LaserScan, gp("scan_topic"), self.scan_callback,
                                 scan_qos, callback_group=self.callback_group)
        self.create_subscription(Path, gp("plan_input_topic"), self.path_callback,
                                 10, callback_group=self.callback_group)
        self.create_subscription(PoseStamped, gp("goal_topic"), self.goal_callback,
                                 10, callback_group=self.callback_group)

        # === [DELAY-COMP] odom订阅（速度反馈） ===
        self.create_subscription(Odometry, gp("odom_topic"), self.odom_callback,
                                 10, callback_group=self.callback_group)

        # ---- 控制定时器：周期固定 = 1/control_frequency（不因补偿改变） ----
        self.control_frequency = self.get_parameter("control_frequency").get_parameter_value().double_value
        if self.control_frequency <= 0:
            raise ValueError(f"Invalid control_frequency: {self.control_frequency}")
        self.control_period = 1.0 / self.control_frequency

        # === [DELAY-COMP] ===
        self.use_delay_compensation = self.get_parameter(
            "use_delay_compensation").get_parameter_value().bool_value
        self.get_logger().info(
            f"Control: {self.control_frequency:.1f} Hz (period {self.control_period*1000:.0f} ms) | "
            f"Delay compensation: {'ON — predict dt=' + f'{self.control_period*1000:.0f}ms' if self.use_delay_compensation else 'OFF'}")

        self.create_timer(self.control_period, self.run, callback_group=self.control_group)

    # ================= [DELAY-COMP] 核心新增 =================
    def odom_callback(self, msg: Odometry) -> None:
        """记录实测速度，用于未来状态估计"""
        with self._state_lock:
            self.current_velocity[0, 0] = msg.twist.twist.linear.x
            self.current_velocity[1, 0] = msg.twist.twist.angular.z
            self.last_odom_time = time.time()

    def _predict_future_state(self, dt: float) -> Optional[npt.NDArray]:
        """需求1: 传入neupan前, 预测 dt=1/control_frequency 秒后的状态。

        速度来源: /odom实测速度（0.5s内有效）优先, 回退到上一条下发指令。
        模型: |w|<1e-6 直线外推, 否则恒转率圆弧外推。
        """
        with self._state_lock:
            if self.robot_state is None:
                return None
            cur = self.robot_state.copy()
            fresh = (self.last_odom_time > 0
                     and (time.time() - self.last_odom_time) < 0.5)
            vel = self.current_velocity.copy() if fresh else self.last_commanded_vel.copy()

        x, y, th = float(cur[0, 0]), float(cur[1, 0]), float(cur[2, 0])
        v, w = float(vel[0, 0]), float(vel[1, 0])

        if abs(w) < 1e-6:
            px = x + v * np.cos(th) * dt
            py = y + v * np.sin(th) * dt
            pth = th
        else:
            r = v / w
            px = x + r * (np.sin(th + w * dt) - np.sin(th))
            py = y - r * (np.cos(th + w * dt) - np.cos(th))
            pth = th + w * dt

        return np.array([[px], [py], [pth]])

    # ================= 官方逻辑（不变） =================
    def _get_robot_transform(self) -> bool:
        try:
            trans = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
            yaw = quat_to_yaw(trans.transform.rotation)
            new_state = np.array([trans.transform.translation.x,
                                  trans.transform.translation.y, yaw]).reshape(3, 1)
            with self._state_lock:
                self.robot_state = new_state
            return True
        except tf2_ros.LookupException:
            self.get_logger().debug("Waiting for TF", throttle_duration_sec=1.0)
            return False
        except tf2_ros.ConnectivityException:
            self.get_logger().warn("TF connectivity", throttle_duration_sec=1.0)
            return False
        except tf2_ros.ExtrapolationException as e:
            self.get_logger().warn(f"TF extrapolation: {e}", throttle_duration_sec=1.0)
            return False

    def _validate_planning_prerequisites(self) -> bool:
        with self._state_lock:
            if self.robot_state is None:
                return False
            if (len(self.neupan_planner.waypoints) >= 1
                    and self.neupan_planner.initial_path is None):
                self.neupan_planner.set_initial_path_from_state(self.robot_state)
            if self.neupan_planner.initial_path is None:
                self.get_logger().debug("Waiting for initial path", throttle_duration_sec=1.0)
                return False
        return True

    def _execute_planning(self, robot_state_override: Optional[npt.NDArray] = None
                          ) -> Tuple[Optional[npt.NDArray], Optional[Dict[str, Any]]]:
        with self._state_lock:
            initial_path = self.neupan_planner.initial_path
        self.ref_path_pub.publish(self.generate_path_msg(initial_path))

        with self._state_lock:
            _plan_state = (robot_state_override if robot_state_override is not None
                           else self.robot_state)
            robot_state_copy = _plan_state.copy() if _plan_state is not None else None
            obstacle_points_copy = (self.obstacle_points.copy()
                                    if self.obstacle_points is not None else None)

        action, info = self.neupan_planner(robot_state_copy, obstacle_points_copy)

        with self._state_lock:
            self.stop = info["stop"]
            self.arrive = info["arrive"]

        if info["arrive"]:
            self.get_logger().info("Arrived at target", once=True)
        if info["stop"]:
            self.get_logger().warn(
                f"Collision risk - min dist: {self.neupan_planner.min_distance:.2f}m",
                throttle_duration_sec=1.0)

        # === [DELAY-COMP] 记录下发指令（预测回退用） ===
        if action is not None:
            with self._state_lock:
                self.last_commanded_vel[0, 0] = float(action[0, 0])
                self.last_commanded_vel[1, 0] = float(action[1, 0])

        return action, info

    # ================= 主循环（改动处已标记） =================
    def run(self) -> None:
        _cycle_start = time.monotonic()   # === [DELAY-COMP] 周期起点

        if not self._get_robot_transform():
            return
        if not self._validate_planning_prerequisites():
            return

        # === [DELAY-COMP] 需求1: 未来状态估计 (dt = 1/control_frequency) ===
        predicted_state = None
        if self.use_delay_compensation:
            predicted_state = self._predict_future_state(self.control_period)

        action, info = self._execute_planning(predicted_state)
        if info is None:
            return

        # 监控类消息立即发（RViz实时性）
        self.plan_pub.publish(self.generate_path_msg(info["opt_state_list"]))
        self.ref_state_pub.publish(self.generate_path_msg(info["ref_state_list"]))
        self.viz_manager.publish_visualization(self.neupan_planner, self.robot_state)

        # === [DELAY-COMP] 需求2: 指令下发对齐控制周期 ===
        # 推理快 → 等到周期边界再发（指令恰好在机器人到达预测位姿时生效）
        # 推理慢 → 立即发并告警（sleep上限=1个周期，不会阻塞回调线程）
        if self.use_delay_compensation:
            _elapsed = time.monotonic() - _cycle_start
            _wait = self.control_period - _elapsed
            if _wait > 0.001:
                time.sleep(_wait)
            elif _wait < -0.005:
                self._overrun_count += 1
                self.get_logger().warn(
                    f"Cycle overrun {-_wait*1000:.0f}ms "
                    f"(solve {_elapsed*1000:.0f}ms > period {self.control_period*1000:.0f}ms), "
                    f"total overruns: {self._overrun_count} — consider lowering control_frequency",
                    throttle_duration_sec=2.0)

        vel_msg = self.generate_twist_msg(action, info["stop"], info["arrive"])
        self.vel_pub.publish(vel_msg)

        _cycle_ms = (time.monotonic() - _cycle_start) * 1000
        self.get_logger().info(
            f"cycle {_cycle_ms:.0f}ms | "
            f"cmd=({float(action[0,0]):.3f}, {float(action[1,0]):.3f})"
            if action is not None else f"cycle {_cycle_ms:.0f}ms | cmd=(0,0)",
            throttle_duration_sec=1.0)

    # ================= 官方回调（不变） =================
    def scan_callback(self, scan_msg: LaserScan) -> Optional[npt.NDArray]:
        with self._state_lock:
            if self.robot_state is None:
                return None

        ranges = np.array(scan_msg.ranges)
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))
        if self.flip_angle:
            angles = np.flip(angles)

        indices = np.arange(len(ranges))
        valid_mask = (((indices % self.scan_downsample) == 0)
                      & (ranges >= self.scan_range[0]) & (ranges <= self.scan_range[1])
                      & (angles > self.scan_angle_range[0]) & (angles < self.scan_angle_range[1]))
        valid_ranges, valid_angles = ranges[valid_mask], angles[valid_mask]

        if len(valid_ranges) == 0:
            with self._state_lock:
                self.obstacle_points = None
            return None

        point_array = np.vstack([valid_ranges * np.cos(valid_angles),
                                 valid_ranges * np.sin(valid_angles)])
        try:
            trans = self.tf_buffer.lookup_transform(
                self.map_frame, self.lidar_frame, rclpy.time.Time())
            yaw = quat_to_yaw(trans.transform.rotation)
            x, y = trans.transform.translation.x, trans.transform.translation.y
            trans_matrix, rot_matrix = get_transform(np.c_[x, y, yaw].reshape(3, 1))
            transformed = rot_matrix @ point_array + trans_matrix
            with self._state_lock:
                self.obstacle_points = transformed
            return transformed
        except tf2_ros.LookupException:
            return None

    def path_callback(self, path: Path) -> None:
        n_poses = len(path.poses)
        if n_poses == 0:
            return
        if self.include_initial_path_direction:
            data = [(p.pose.position.x, p.pose.position.y, quat_to_yaw(p.pose.orientation))
                    for p in path.poses]
            xs, ys, thetas = np.array(data).T
        else:
            coords = [(p.pose.position.x, p.pose.position.y) for p in path.poses]
            xs, ys = np.array(coords).T
            dx = np.diff(xs, append=xs[-1])
            dy = np.diff(ys, append=ys[-1])
            thetas = np.arctan2(dy, dx)
            if n_poses > 1:
                thetas[-1] = thetas[-2]
        initial_point_array = np.vstack([xs, ys, thetas, np.ones(n_poses)])
        initial_point_list = [initial_point_array[:, i:i + 1] for i in range(n_poses)]
        with self._state_lock:
            if (self.neupan_planner.initial_path is None or self.refresh_initial_path):
                self.neupan_planner.set_initial_path(initial_point_list)

    def goal_callback(self, goal: PoseStamped) -> None:
        x, y = goal.pose.position.x, goal.pose.position.y
        theta = quat_to_yaw(goal.pose.orientation)
        new_goal = np.array([[x], [y], [theta]])
        self.get_logger().info(f"New goal: ({x:.2f}, {y:.2f}, {theta:.2f})")
        if self.robot_state is None:
            self.goal = new_goal
            return
        with self._state_lock:
            self.goal = new_goal
            self.neupan_planner.update_initial_path_from_goal(self.robot_state, self.goal)
            self.neupan_planner.reset()

    def generate_path_msg(self, path_list: List[npt.NDArray]) -> Path:
        path = Path()
        path.header.frame_id = self.map_frame
        path.header.stamp = self.get_clock().now().to_msg()
        if len(path_list) == 0:
            return path
        normalized = []
        for point in path_list:
            arr = np.array(point)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            normalized.append(arr[:3, :])
        pm = np.hstack(normalized)
        for x, y, yaw in zip(pm[0, :], pm[1, :], pm[2, :]):
            ps = PoseStamped()
            ps.header.frame_id = self.map_frame
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation = yaw_to_quat(float(yaw))
            path.poses.append(ps)
        return path

    def generate_twist_msg(self, vel: Optional[npt.NDArray], stop: bool, arrive: bool) -> Twist:
        if vel is None or stop or arrive:
            return Twist()
        action = Twist()
        action.linear.x = float(vel[0, 0])
        action.angular.z = float(vel[1, 0])
        return action


def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = None
    try:
        node = NeupanCore()
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        node.get_logger().info(
            f"NeuPAN node started - Control: {node.control_frequency}Hz, "
            f"DelayComp: {node.use_delay_compensation}")
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node:
            node.get_logger().error(f'Unhandled: {e}\n{traceback.format_exc()}')
        raise
    finally:
        if executor:
            executor.shutdown()
        if node:
            node.vel_pub.publish(Twist())
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
