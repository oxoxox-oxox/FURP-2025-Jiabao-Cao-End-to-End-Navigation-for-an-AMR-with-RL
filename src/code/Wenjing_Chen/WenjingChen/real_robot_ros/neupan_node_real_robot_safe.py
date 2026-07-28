#!/usr/bin/env python

"""
NeupanCore is the main ROS2 node for the NeuPAN navigation algorithm.

This node subscribes to laser scan and localization data, executes the NeuPAN
planning algorithm, and publishes velocity commands to control the robot.

Developer: Han Ruihua <hanrh@connect.hku.hk>  Li Chengyang <kevinladlee@gmail.com>
Date: 2025.04.08

Latency-aware real-robot revision (neupan_node_real_robot_safe.py)
-----------------------------------------------------------------
相对原始 neupan_node.py 的延迟处理原则（避免 TF 时间错位造成真机导航劣化）:

1. 点云必须用 scan 时间戳上的 TF 变换到 map；默认不 fallback 到 latest
   （latest 可能比 scan 旧 1~3 秒，会让障碍物坐标和机器人状态严重错位）
2. 若 scan-stamp TF 暂时不可用，则跳过该帧扫描；旧障碍超过 max_scan_age 后
   会被丢弃，可选 stop_on_stale_scan 让真机安全停车
3. 规划用“当前” TF 位姿；只用里程计实测 twist (v,w) 做短时外推，不用 cmd_hist
4. 对输出 cmd 做轻度一阶低通 + 加速度限幅，抑制绕障后回参考线时的抖振
5. 保留原始 NeuPAN 控制/规划/可视化/多线程结构，不重写整条流水线

对比实验开关:
  compensate_delay:=false  → 仅时间戳同步点云，不做状态外推（最接近原版）
  cmd_smoothing:=false     → 关闭输出滤波/限加速度
  allow_latest_tf_fallback:=true → 允许退回 latest TF（仅建议临时排查）
"""
import math
import os
import threading
import time
import traceback
from collections import deque
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import numpy.typing as npt
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.time import Time
from ament_index_python.packages import get_package_share_directory
import tf2_ros

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path, Odometry
from sensor_msgs.msg import LaserScan

try:
    from neupan import neupan
    from neupan.util import get_transform
except ImportError as e:
    raise ImportError(
        f"Failed to import 'neupan' package: {e}. "
        "Please install NeuPAN first."
    ) from e

# Import local modules
from neupan_ros2.visualization_manager import VisualizationManager
from neupan_ros2.utils import yaw_to_quat, quat_to_yaw


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


class NeupanCore(Node):
    """ROS2 node for NeuPAN navigation algorithm.

    This node integrates the NeuPAN planner with ROS2, handling sensor data,
    executing planning, and publishing control commands.
    """

    def __init__(self) -> None:
        super().__init__("neupan_node")

        # Thread lock protecting shared state: robot_state, obstacle_points, stop, arrive
        # These are accessed by both control thread (run) and callback thread (scan/path/goal)
        self._state_lock = threading.Lock()

        # Callback groups for multi-threaded execution
        # Control group: MutuallyExclusive for timer (run) - ensures run() executes alone
        self.control_group = MutuallyExclusiveCallbackGroup()
        # Callback group: Reentrant for all subscriptions - allows concurrent execution
        self.callback_group = ReentrantCallbackGroup()

        # Package directory for accessing config files and models
        self.pkg_dir = get_package_share_directory("neupan_ros2")

        # Robot identification and configuration directory
        self.declare_parameter("robot_type", "")
        self.declare_parameter("robot_description", "")
        self.declare_parameter("robot_config_dir", "")  # Set by launch file
        self.declare_parameter("planner_config_file", "planner.yaml")
        self.declare_parameter("dune_checkpoint_file", "models/dune_model_5000.pth")

        # Legacy parameter name (for backward compatibility)
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
        self.declare_parameter("control_frequency", 50.0)  # Control loop frequency in Hz

        # Visualization control parameters
        self.declare_parameter("enable_visualization", True)
        self.declare_parameter("enable_dune_markers", True)
        self.declare_parameter("enable_nrmp_markers", True)
        self.declare_parameter("enable_robot_marker", True)

        # Topic names (configurable for flexibility)
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

        # === Real-robot latency / smoothness parameters ===
        # 用实测 odom twist 短时外推，替代 cmd_hist 开环 rollforward
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("compensate_delay", True)
        self.declare_parameter("max_odom_age", 0.35)
        # 电机/底盘执行滞后（命令发出 → 实际速度跟上），建议 0.05~0.20s
        self.declare_parameter("actuation_delay", 0.08)
        # 外推时间上限，防止长时间开环漂移；超过则钳位并打日志
        self.declare_parameter("max_predict_horizon", 0.35)
        # TF 查历史时允许等待的时间；真机 TF 常比 scan 晚几十毫秒到达
        self.declare_parameter("tf_lookup_timeout", 0.12)
        # 默认不允许 scan-stamp TF 失败后退回 latest；latest 过旧会直接拉坏点云
        self.declare_parameter("allow_latest_tf_fallback", False)
        # 障碍点云最大可用年龄；超过后不再参与规划
        self.declare_parameter("max_scan_age", 0.45)
        # 点云/TF 过期时是否安全停车，而不是继续沿路径跑
        self.declare_parameter("stop_on_stale_scan", True)
        # 输出平滑：一阶低通 alpha∈(0,1]，越小越平滑；1=关闭低通
        self.declare_parameter("cmd_smoothing", True)
        self.declare_parameter("cmd_lpf_alpha", 0.35)
        # 加速度限幅 (m/s^2, rad/s^2)；0 表示不限制
        self.declare_parameter("max_lin_acc", 0.8)
        self.declare_parameter("max_ang_acc", 3.0)
        # 求解很慢时不把整段耗时都拿去外推；长外推比少补偿更危险
        self.declare_parameter("max_solve_delay_for_prediction", 0.16)
        # 周期延迟统计日志
        self.declare_parameter("delay_report_period", 5.0)

        # === Configuration Loading ===
        # Get robot configuration directory (set by launch file)
        robot_config_dir = (
            self.get_parameter("robot_config_dir")
            .get_parameter_value().string_value
        )

        # Validate robot config directory exists
        if not robot_config_dir or not os.path.isdir(robot_config_dir):
            raise ValueError(
                f"Invalid robot_config_dir: '{robot_config_dir}'. "
                "Must be set by launch file to a valid robot config directory."
            )

        # Get robot type for logging
        robot_type = (
            self.get_parameter("robot_type")
            .get_parameter_value().string_value
        )
        robot_description = (
            self.get_parameter("robot_description")
            .get_parameter_value().string_value
        )

        self.get_logger().info(f"Loading robot configuration: {robot_type}")
        self.get_logger().info(f"Description: {robot_description}")
        self.get_logger().info(f"Config directory: {robot_config_dir}")

        # Load planner configuration (relative to robot config dir)
        planner_config_file = (
            self.get_parameter("planner_config_file")
            .get_parameter_value().string_value
        )
        self.planner_config_file = os.path.join(robot_config_dir, planner_config_file)

        # Load DUNE checkpoint (relative to robot config dir)
        dune_checkpoint_file = (
            self.get_parameter("dune_checkpoint_file")
            .get_parameter_value().string_value
        )
        self.dune_checkpoint = os.path.join(robot_config_dir, dune_checkpoint_file)

        # Validate configuration files exist
        if not os.path.isfile(self.planner_config_file):
            raise FileNotFoundError(
                f"Planner config not found: {self.planner_config_file}"
            )
        if not os.path.isfile(self.dune_checkpoint):
            raise FileNotFoundError(
                f"DUNE checkpoint not found: {self.dune_checkpoint}"
            )

        self.get_logger().info(f"Planner config: {self.planner_config_file}")
        self.get_logger().info(f"DUNE checkpoint: {self.dune_checkpoint}")

        # Load other parameters
        self.map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.lidar_frame = self.get_parameter("lidar_frame").get_parameter_value().string_value
        self.marker_size = self.get_parameter("marker_size").get_parameter_value().double_value
        self.marker_z = self.get_parameter("marker_z").get_parameter_value().double_value

        self.scan_range = np.array([
            self.get_parameter("scan_range_min").get_parameter_value().double_value,
            self.get_parameter("scan_range_max").get_parameter_value().double_value
        ])

        self.scan_angle_range = np.array([
            self.get_parameter("scan_angle_min").get_parameter_value().double_value,
            self.get_parameter("scan_angle_max").get_parameter_value().double_value
        ])

        self.scan_downsample = (
            self.get_parameter("scan_downsample")
            .get_parameter_value().integer_value
        )

        self.refresh_initial_path = (
            self.get_parameter("refresh_initial_path")
            .get_parameter_value().bool_value
        )
        self.flip_angle = (
            self.get_parameter("flip_angle")
            .get_parameter_value().bool_value
        )
        self.include_initial_path_direction = (
            self.get_parameter("include_initial_path_direction")
            .get_parameter_value().bool_value
        )

        self.enable_visualization = (
            self.get_parameter("enable_visualization")
            .get_parameter_value().bool_value
        )
        self.enable_dune_markers = (
            self.get_parameter("enable_dune_markers")
            .get_parameter_value().bool_value
        )
        self.enable_nrmp_markers = (
            self.get_parameter("enable_nrmp_markers")
            .get_parameter_value().bool_value
        )
        self.enable_robot_marker = (
            self.get_parameter("enable_robot_marker")
            .get_parameter_value().bool_value
        )

        # Latency / smoothing params
        self.compensate_delay = (
            self.get_parameter("compensate_delay").get_parameter_value().bool_value
        )
        self.max_odom_age = (
            self.get_parameter("max_odom_age").get_parameter_value().double_value
        )
        self.actuation_delay = (
            self.get_parameter("actuation_delay").get_parameter_value().double_value
        )
        self.max_predict_horizon = (
            self.get_parameter("max_predict_horizon").get_parameter_value().double_value
        )
        self.tf_lookup_timeout = (
            self.get_parameter("tf_lookup_timeout").get_parameter_value().double_value
        )
        self.allow_latest_tf_fallback = (
            self.get_parameter("allow_latest_tf_fallback")
            .get_parameter_value().bool_value
        )
        self.max_scan_age = (
            self.get_parameter("max_scan_age").get_parameter_value().double_value
        )
        self.stop_on_stale_scan = (
            self.get_parameter("stop_on_stale_scan").get_parameter_value().bool_value
        )
        self.cmd_smoothing = (
            self.get_parameter("cmd_smoothing").get_parameter_value().bool_value
        )
        self.cmd_lpf_alpha = (
            self.get_parameter("cmd_lpf_alpha").get_parameter_value().double_value
        )
        self.max_lin_acc = (
            self.get_parameter("max_lin_acc").get_parameter_value().double_value
        )
        self.max_ang_acc = (
            self.get_parameter("max_ang_acc").get_parameter_value().double_value
        )
        self.max_solve_delay_for_prediction = (
            self.get_parameter("max_solve_delay_for_prediction")
            .get_parameter_value().double_value
        )
        delay_report_period = (
            self.get_parameter("delay_report_period").get_parameter_value().double_value
        )

        if self.refresh_initial_path:
            self.get_logger().info("Refresh initial path is enabled")

        if not self.planner_config_file:
            raise ValueError(
                "No planner config file provided! "
                "Please set the parameter 'config_file'"
            )

        pan = {'dune_checkpoint': self.dune_checkpoint}
        self.neupan_planner = neupan.init_from_yaml(self.planner_config_file, pan=pan)

        # Log robot dimensions for verification
        self.get_logger().info(
            f"Robot dimensions - Length: {self.neupan_planner.robot.length:.3f}m, "
            f"Width: {self.neupan_planner.robot.width:.3f}m"
        )
        if hasattr(self.neupan_planner.robot, 'wheelbase') and self.neupan_planner.robot.wheelbase is not None:
            self.get_logger().info(
                f"Robot wheelbase: {self.neupan_planner.robot.wheelbase:.3f}m"
            )
        self.get_logger().info(f"Robot kinematics: {self.neupan_planner.robot.kinematics}")
        self.get_logger().info("NeuPAN planner initialized successfully")
        self.get_logger().info(
            f"Latency compensate={self.compensate_delay}, "
            f"actuation_delay={self.actuation_delay:.3f}s, "
            f"max_predict={self.max_predict_horizon:.3f}s, "
            f"cmd_smoothing={self.cmd_smoothing}, "
            f"latest_tf_fallback={self.allow_latest_tf_fallback}, "
            f"max_scan_age={self.max_scan_age:.3f}s"
        )

        # Shared state protected by _state_lock (accessed by multiple threads)
        # Write access: scan_callback (obstacle_points), _get_robot_transform (robot_state)
        # Read access: _execute_planning (all), generate_twist_msg (stop, arrive)
        # Planning copies data before execution to minimize lock holding time
        self.obstacle_points: Optional[npt.NDArray] = None  # (2, n) obstacle points in map frame
        self.obstacle_stamp: Optional[float] = None  # stamp of TF/scan used for obstacle_points
        self.robot_state: Optional[npt.NDArray] = None  # (3, 1) [x, y, theta] in map frame
        self.stop: bool = False  # Emergency stop flag from collision detection
        self.arrive: bool = False  # Goal reached flag
        self.goal: Optional[npt.NDArray] = None  # (3, 1) target goal [x, y, theta]
        self.sensor_stale: bool = False  # True when no fresh scan-stamp TF/points are available

        # --- Latency state (not using published-command history) ---
        self._odom_twist = (0.0, 0.0)  # measured (v, w) from /odom
        self._odom_stamp: Optional[float] = None
        self._solve_ema: float = 0.05  # planner solve-time EMA (s)
        self._stat_scan_age: deque = deque(maxlen=300)
        self._stat_predict: deque = deque(maxlen=300)
        self._stat_solve: deque = deque(maxlen=300)
        # Smoothed command state for LPF / rate limit
        self._cmd_filt_v: float = 0.0
        self._cmd_filt_w: float = 0.0
        self._last_cmd_time: Optional[float] = None

        self.vel_pub = self.create_publisher(
            Twist,
            self.get_parameter("cmd_vel_topic").get_parameter_value().string_value,
            10
        )
        self.plan_pub = self.create_publisher(
            Path,
            self.get_parameter("plan_output_topic").get_parameter_value().string_value,
            10
        )
        self.ref_state_pub = self.create_publisher(
            Path,
            self.get_parameter("ref_state_topic").get_parameter_value().string_value,
            10
        )
        self.ref_path_pub = self.create_publisher(
            Path,
            self.get_parameter("initial_path_topic").get_parameter_value().string_value,
            10
        )

        # Initialize visualization manager (handles all visualization independently)
        viz_config = {
            'enable_visualization': self.enable_visualization,
            'enable_dune_markers': self.enable_dune_markers,
            'enable_nrmp_markers': self.enable_nrmp_markers,
            'enable_robot_marker': self.enable_robot_marker,
            'map_frame': self.map_frame,
            'marker_size': self.marker_size,
            'marker_z': self.marker_z,
            'dune_markers_topic': (
                self.get_parameter("dune_markers_topic")
                .get_parameter_value().string_value
            ),
            'nrmp_markers_topic': (
                self.get_parameter("nrmp_markers_topic")
                .get_parameter_value().string_value
            ),
            'robot_marker_topic': (
                self.get_parameter("robot_marker_topic")
                .get_parameter_value().string_value
            ),
            'state_lock': self._state_lock
        }
        self.viz_manager = VisualizationManager(self, viz_config)

        # TF listener for coordinate transformations (default 10s buffer)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        scan_qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(
            LaserScan,
            self.get_parameter("scan_topic").get_parameter_value().string_value,
            self.scan_callback,
            scan_qos_profile,
            callback_group=self.callback_group
        )
        self.create_subscription(
            Path,
            self.get_parameter("plan_input_topic").get_parameter_value().string_value,
            self.path_callback,
            10,
            callback_group=self.callback_group
        )
        self.create_subscription(
            PoseStamped,
            self.get_parameter("goal_topic").get_parameter_value().string_value,
            self.goal_callback,
            10,
            callback_group=self.callback_group
        )
        # 里程计：仅用实测速度做短时外推（不用 cmd_hist）
        odom_topic = (
            self.get_parameter("odom_topic").get_parameter_value().string_value
        )
        self.create_subscription(
            Odometry,
            odom_topic,
            self.odom_callback,
            20,
            callback_group=self.callback_group
        )

        # Control loop timer: frequency configurable via parameter
        self.control_frequency = (
            self.get_parameter("control_frequency")
            .get_parameter_value().double_value
        )
        if self.control_frequency <= 0:
            raise ValueError(
                f"Invalid control_frequency: {self.control_frequency}. "
                "Must be > 0 Hz"
            )

        time_period = 1.0 / self.control_frequency
        self.get_logger().info(
            f"Control loop frequency: {self.control_frequency} Hz "
            f"({time_period*1000:.1f} ms period)"
        )
        self.create_timer(time_period, self.run, callback_group=self.control_group)

        if delay_report_period > 0:
            self.create_timer(
                delay_report_period,
                self._report_delay_stats,
                callback_group=self.control_group,
            )

    # ------------------------------------------------------------------
    # Latency helpers
    # ------------------------------------------------------------------
    def _stamp_to_sec(self, stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _now_sec(self) -> float:
        t = self.get_clock().now().to_msg()
        return self._stamp_to_sec(t)

    def _transform_stamp_sec(self, trans) -> float:
        return self._stamp_to_sec(trans.header.stamp)

    def _stamp_is_zero(self, stamp) -> bool:
        return stamp.sec == 0 and stamp.nanosec == 0

    def odom_callback(self, msg: Odometry) -> None:
        """Cache measured body twist from odometry (not commanded velocity)."""
        v = float(msg.twist.twist.linear.x)
        w = float(msg.twist.twist.angular.z)
        st = self._stamp_to_sec(msg.header.stamp)
        with self._state_lock:
            self._odom_twist = (v, w)
            self._odom_stamp = st

    def _lookup_transform_at(self, target_frame: str, source_frame: str, stamp_msg):
        """Lookup TF at a message stamp.

        默认不退回 latest。真机上 latest TF 若明显落后 scan，会把障碍点云放到
        错误的世界位置，比跳过这一帧更危险。
        """
        timeout = Duration(seconds=self.tf_lookup_timeout)
        requested_sec = self._stamp_to_sec(stamp_msg)
        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time.from_msg(stamp_msg),
                timeout=timeout,
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            if not self.allow_latest_tf_fallback:
                self.get_logger().warn(
                    f"Skip scan: TF@{source_frame}->{target_frame} at scan stamp "
                    f"is unavailable ({e})",
                    throttle_duration_sec=1.0,
                )
                raise

            latest = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),  # latest
                timeout=timeout,
            )
            latest_sec = self._transform_stamp_sec(latest)
            tf_lag = max(0.0, requested_sec - latest_sec)
            if tf_lag > self.max_scan_age:
                self.get_logger().warn(
                    f"Skip scan: latest TF is too old for scan "
                    f"(lag={tf_lag*1000:.0f}ms > {self.max_scan_age*1000:.0f}ms)",
                    throttle_duration_sec=1.0,
                )
                raise

            self.get_logger().warn(
                f"Using latest TF fallback for scan "
                f"(lag={tf_lag*1000:.0f}ms)",
                throttle_duration_sec=1.0,
            )
            return latest

    def _predict_pose_diff_drive(
        self, pose: npt.NDArray, v: float, w: float, dt: float
    ) -> npt.NDArray:
        """Short-horizon unicycle prediction using measured (v, w).

        仅用于短暂延迟补偿。转弯用精确圆弧积分，避免欧拉角误差放大位置偏差。
        故意不用发布的 cmd_hist：命令≠实际速度，长历史开环积分会在转弯时抖。
        """
        if dt <= 0.0:
            return pose.copy()

        x = float(pose[0, 0])
        y = float(pose[1, 0])
        th = float(pose[2, 0])

        if abs(w) < 1e-6:
            # Nearly straight: Euler is fine and stable
            x += v * math.cos(th) * dt
            y += v * math.sin(th) * dt
            th += w * dt
        else:
            # Exact circular-arc integration for constant (v, w)
            th_new = th + w * dt
            x += (v / w) * (math.sin(th_new) - math.sin(th))
            y += (v / w) * (-math.cos(th_new) + math.cos(th))
            th = th_new

        out = np.array([[x], [y], [_wrap_angle(th)]], dtype=float)
        return out

    def _compensate_state_for_delay(self, pose_now: npt.NDArray) -> npt.NDArray:
        """Predict pose at approximate command-apply time using odom twist.

        total_delay ≈ (now - scan_stamp) 的一部分不必再外推（障碍已按扫描时刻
        正确落在 map，当前 TF 已包含“扫描后到现在”的运动）；只需补偿：
          remaining = solve_ema + actuation_delay
        即从“现在”再往前推一点到动作生效。若想与观测龄更保守对齐，也可加上
        少量 scan_age 的比例项，但默认只用求解+执行延迟，避免过大开环。
        """
        if not self.compensate_delay:
            return pose_now

        with self._state_lock:
            v, w = self._odom_twist
            odom_stamp = self._odom_stamp

        if odom_stamp is None:
            self._stat_predict.append(0.0)
            return pose_now

        odom_age = self._now_sec() - odom_stamp
        if odom_age > self.max_odom_age:
            self.get_logger().warn(
                f"Skip delay prediction: odom twist is stale "
                f"({odom_age*1000:.0f}ms > {self.max_odom_age*1000:.0f}ms)",
                throttle_duration_sec=1.0,
            )
            self._stat_predict.append(0.0)
            return pose_now

        # 规划求解耗时 EMA + 机电感测延迟；不把整段 scan_age 再外推一遍
        # （当前 TF 位姿已是“现在”，点云已用 scan-time TF）
        solve_delay = max(0.0, self._solve_ema)
        if solve_delay > self.max_solve_delay_for_prediction:
            self.get_logger().warn(
                f"Solve EMA {solve_delay*1000:.0f}ms exceeds prediction budget; "
                f"using {self.max_solve_delay_for_prediction*1000:.0f}ms",
                throttle_duration_sec=2.0,
            )
            solve_delay = self.max_solve_delay_for_prediction

        dt = max(0.0, solve_delay + self.actuation_delay)
        if dt > self.max_predict_horizon:
            self.get_logger().warn(
                f"Predict horizon {dt*1000:.0f}ms clamped to "
                f"{self.max_predict_horizon*1000:.0f}ms",
                throttle_duration_sec=2.0,
            )
            dt = self.max_predict_horizon

        self._stat_predict.append(dt)

        return self._predict_pose_diff_drive(pose_now, v, w, dt)

    def _smooth_cmd(self, v: float, w: float) -> Tuple[float, float]:
        """Low-pass + acceleration limit on published commands."""
        if not self.cmd_smoothing:
            return v, w

        now = self._now_sec()
        if self._last_cmd_time is None:
            dt = 1.0 / max(self.control_frequency, 1.0)
        else:
            dt = max(1e-3, now - self._last_cmd_time)
        self._last_cmd_time = now

        # First-order low-pass
        a = float(np.clip(self.cmd_lpf_alpha, 0.05, 1.0))
        v_f = a * v + (1.0 - a) * self._cmd_filt_v
        w_f = a * w + (1.0 - a) * self._cmd_filt_w

        # Rate / acceleration limiting (optional)
        if self.max_lin_acc > 0.0:
            dv_max = self.max_lin_acc * dt
            v_f = float(np.clip(v_f, self._cmd_filt_v - dv_max, self._cmd_filt_v + dv_max))
        if self.max_ang_acc > 0.0:
            dw_max = self.max_ang_acc * dt
            w_f = float(np.clip(w_f, self._cmd_filt_w - dw_max, self._cmd_filt_w + dw_max))

        self._cmd_filt_v, self._cmd_filt_w = v_f, w_f
        return v_f, w_f

    def _report_delay_stats(self) -> None:
        if not self._stat_scan_age and not self._stat_solve:
            return
        age = np.array(self._stat_scan_age) * 1000.0 if self._stat_scan_age else np.array([0.0])
        sol = np.array(self._stat_solve) * 1000.0 if self._stat_solve else np.array([0.0])
        pred = np.array(self._stat_predict) * 1000.0 if self._stat_predict else np.array([0.0])
        self.get_logger().info(
            f"[delay] scan_age {age.mean():.0f}/{age.max():.0f}ms | "
            f"solve {sol.mean():.0f}/{sol.max():.0f}ms | "
            f"predict {pred.mean():.0f}ms | "
            f"comp={self.compensate_delay} smooth={self.cmd_smoothing}"
        )

    def _get_robot_transform(self) -> bool:
        """Get robot transform from TF and update robot_state.

        Returns:
            bool: True if transform successfully obtained, False otherwise

        """
        try:
            # TF query is thread-safe, no lock needed
            # Use latest transform for "current" robot pose (planning reference).
            trans = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, Time()
            )

            yaw = quat_to_yaw(trans.transform.rotation)
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            new_state = np.array([x, y, yaw]).reshape(3, 1)

            # Optional short forward prediction with measured odom twist.
            new_state = self._compensate_state_for_delay(new_state)

            # Lock only for writing shared state
            with self._state_lock:
                self.robot_state = new_state

            self.get_logger().info(
                f"Robot state initialized - x: {new_state[0,0]:.2f}m, "
                f"y: {new_state[1,0]:.2f}m, yaw: {new_state[2,0]:.2f}rad",
                once=True
            )
            return True

        except tf2_ros.LookupException:
            self.get_logger().debug(
                f"Waiting for transform from {self.base_frame} to {self.map_frame}",
                throttle_duration_sec=1.0,
            )
            return False
        except tf2_ros.ConnectivityException:
            self.get_logger().warn(
                "ConnectivityException: Transform not available, waiting for connection",
                throttle_duration_sec=1.0
            )
            return False
        except tf2_ros.ExtrapolationException as e:
            self.get_logger().warn(
                f"TF extrapolation error: {e}. Check TF timestamps and buffer size.",
                throttle_duration_sec=1.0
            )
            return False

    def _validate_planning_prerequisites(self) -> bool:
        """Validate all prerequisites for planning are met.

        Returns:
            bool: True if all prerequisites are met, False otherwise

        """
        with self._state_lock:
            if self.robot_state is None:
                self.get_logger().debug("Waiting for robot state", throttle_duration_sec=1.0)
                return False

            # Initialize path from waypoints on first run if no path received
            if (len(self.neupan_planner.waypoints) >= 1
                    and self.neupan_planner.initial_path is None):
                self.neupan_planner.set_initial_path_from_state(
                    self.robot_state
                )
                self.get_logger().info(
                    f'Initialized path with '
                    f'{len(self.neupan_planner.waypoints)} waypoints'
                )

            if self.neupan_planner.initial_path is None:
                self.get_logger().debug("Waiting for initial path", throttle_duration_sec=1.0)
                return False

        return True

    def _execute_planning(self) -> Tuple[Optional[npt.NDArray], Dict[str, Any]]:
        """Execute planning and update state.

        Returns:
            tuple: (action, info) from neupan planner, or (None, None) on failure

        """
        # Publish reference path (generate message needs to read planner state)
        with self._state_lock:
            initial_path = self.neupan_planner.initial_path

        # Publishing is thread-safe, do outside lock
        self.ref_path_pub.publish(self.generate_path_msg(initial_path))

        # Step 1: Fast data copy inside lock to minimize lock holding time
        with self._state_lock:
            # Copy state data for planning
            # (allows other threads to access shared state)
            robot_state_copy = (
                self.robot_state.copy()
                if self.robot_state is not None else None
            )
            obstacle_points_copy = (
                self.obstacle_points.copy()
                if self.obstacle_points is not None else None
            )
            obstacle_stamp_copy = self.obstacle_stamp

            # Check for obstacles
            has_obstacles = obstacle_points_copy is not None

        scan_age = None
        sensor_stale = obstacle_stamp_copy is None
        if obstacle_stamp_copy is not None:
            scan_age = max(0.0, self._now_sec() - obstacle_stamp_copy)
            self._stat_scan_age.append(scan_age)
            if scan_age > self.max_scan_age:
                obstacle_points_copy = None
                has_obstacles = False
                sensor_stale = True
                self.get_logger().warn(
                    f"Drop stale scan/obstacles: age={scan_age*1000:.0f}ms "
                    f"> {self.max_scan_age*1000:.0f}ms",
                    throttle_duration_sec=1.0,
                )

        # Step 2: Execute planning OUTSIDE lock (10-50ms)
        # (allows other threads to access shared state)
        t0 = time.monotonic()
        action, info = self.neupan_planner(
            robot_state_copy, obstacle_points_copy
        )
        solve = time.monotonic() - t0
        self._stat_solve.append(solve)
        # EMA of solve time feeds next-cycle delay compensation
        self._solve_ema = 0.9 * self._solve_ema + 0.1 * solve

        # Step 3: Write back results inside lock (< 0.1 μs)
        with self._state_lock:
            self.stop = info["stop"]
            self.arrive = info["arrive"]
            self.sensor_stale = sensor_stale

        # Logging outside lock
        if not has_obstacles:
            self.get_logger().info(
                "No obstacle points detected, performing path tracking only",
                throttle_duration_sec=1.0,
            )

        if sensor_stale and self.stop_on_stale_scan:
            self.get_logger().warn(
                "Fresh scan-stamp TF/obstacle data unavailable; publishing zero cmd",
                throttle_duration_sec=1.0,
            )

        # Log arrival
        if info["arrive"]:
            self.get_logger().info("Arrived at target", once=True)

        # Log stop condition
        if info["stop"]:
            # Read min_distance and threshold outside lock
            # (assume read-only access is safe)
            self.get_logger().warn(
                f"Collision risk detected - "
                f"min distance: {self.neupan_planner.min_distance:.2f}m, "
                f"threshold: {self.neupan_planner.collision_threshold:.2f}m",
                throttle_duration_sec=1.0,
            )

        return action, info

    def _publish_planning_results(
            self, action: Optional[npt.NDArray], info: Dict[str, Any]
    ) -> None:
        """Publish planning results and visualization markers.

        Args:
            action: Control action from planner
            info: Planning info dictionary

        """
        # Publish path messages (info is local, thread-safe)
        self.plan_pub.publish(self.generate_path_msg(info["opt_state_list"]))
        self.ref_state_pub.publish(self.generate_path_msg(info["ref_state_list"]))

        # Generate twist message using info dict (avoid reading shared state)
        with self._state_lock:
            sensor_stale = self.sensor_stale
        stop_for_safety = info["stop"] or (
            self.stop_on_stale_scan and sensor_stale
        )
        vel_msg = self.generate_twist_msg(action, stop_for_safety, info["arrive"])
        self.vel_pub.publish(vel_msg)

        # Visualization (delegated to visualization manager)
        self.viz_manager.publish_visualization(
            self.neupan_planner, self.robot_state
        )

    def run(self) -> None:
        """Execute main control loop at fixed frequency.

        Note: Fine-grained locking is handled within each helper method.
        """
        # Step 1: Get robot transform (locks internally for robot_state write)
        if not self._get_robot_transform():
            return

        # Step 2: Validate planning prerequisites (locks internally for state read)
        if not self._validate_planning_prerequisites():
            return

        # Step 3: Execute planning (locks internally for planning execution)
        action, info = self._execute_planning()

        # Step 4: Publish results (locks internally for marker generation)
        self._publish_planning_results(action, info)

    def scan_callback(self, scan_msg: LaserScan) -> Optional[npt.NDArray]:
        """Process laser scan data and update obstacle points in map frame.

        Args:
            scan_msg: LaserScan message from sensor

        Returns:
            Transformed obstacle points or None if processing failed

        """
        # Quick check if robot state is available (lock briefly)
        scan_has_stamp = not self._stamp_is_zero(scan_msg.header.stamp)
        scan_stamp_sec = (
            self._stamp_to_sec(scan_msg.header.stamp)
            if scan_has_stamp else self._now_sec()
        )
        with self._state_lock:
            if self.robot_state is None:
                return None

        ranges = np.array(scan_msg.ranges)
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))

        if self.flip_angle:
            angles = np.flip(angles)

        # Vectorized filtering: Apply downsampling, range, and angle constraints
        indices = np.arange(len(ranges))
        downsample_mask = (indices % self.scan_downsample) == 0
        range_mask = (ranges >= self.scan_range[0]) & (ranges <= self.scan_range[1])
        angle_mask = (angles > self.scan_angle_range[0]) & (angles < self.scan_angle_range[1])

        valid_mask = downsample_mask & range_mask & angle_mask
        valid_ranges = ranges[valid_mask]
        valid_angles = angles[valid_mask]

        if len(valid_ranges) == 0:
            # Fresh scan with no usable obstacle points in the configured window.
            with self._state_lock:
                self.obstacle_points = None
                self.obstacle_stamp = scan_stamp_sec
                self.sensor_stale = False
            self.get_logger().warn(
                "No valid scan points after filtering",
                throttle_duration_sec=1.0
            )
            return None

        # Vectorized coordinate computation (faster than loop)
        x_coords = valid_ranges * np.cos(valid_angles)
        y_coords = valid_ranges * np.sin(valid_angles)
        point_array = np.vstack([x_coords, y_coords])

        try:
            # CRITICAL: transform obstacles with TF at scan stamp, not "latest".
            # 用扫描时刻位姿把点云放入 map，否则绕障后回参考线时障碍/状态错位会放大抖振。
            if scan_has_stamp:
                trans = self._lookup_transform_at(
                    self.map_frame, self.lidar_frame, scan_msg.header.stamp
                )
            else:
                self.get_logger().warn(
                    "LaserScan has zero stamp; using latest TF for this scan",
                    throttle_duration_sec=2.0,
                )
                trans = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    self.lidar_frame,
                    Time(),
                    timeout=Duration(seconds=self.tf_lookup_timeout),
                )

            yaw = quat_to_yaw(trans.transform.rotation)
            x = trans.transform.translation.x
            y = trans.transform.translation.y

            trans_matrix, rot_matrix = get_transform(np.c_[x, y, yaw].reshape(3, 1))
            transformed_points = rot_matrix @ point_array + trans_matrix

            # Lock only for writing shared state
            with self._state_lock:
                self.obstacle_points = transformed_points
                self.obstacle_stamp = scan_stamp_sec
                self.sensor_stale = False

            self.get_logger().info(
                f"Laser scan initialized with {transformed_points.shape[1]} "
                "points", once=True
            )
            return transformed_points

        except tf2_ros.LookupException:
            self.get_logger().debug(
                f"Waiting for transform from {self.lidar_frame} to {self.map_frame}",
                throttle_duration_sec=1.0
            )
            return
        except (tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().debug(
                f"Scan skipped because scan-stamp TF is not usable: {e}",
                throttle_duration_sec=1.0,
            )
            return

    def path_callback(self, path: Path) -> None:
        """Update initial path from received path message.

        Args:
            path: Path message containing waypoints

        """
        n_poses = len(path.poses)
        if n_poses == 0:
            return

        self.get_logger().info(f"Received new path with {n_poses} waypoints")

        # Optimized: single-pass extraction with transpose
        if self.include_initial_path_direction:
            # Extract x, y, and orientation in one pass
            data = [
                (p.pose.position.x, p.pose.position.y,
                 quat_to_yaw(p.pose.orientation))
                for p in path.poses
            ]
            xs, ys, thetas = np.array(data).T
        else:
            self.get_logger().debug(
                "Using path gradient for direction "
                "(include_initial_path_direction=False)", once=True
            )

            # Extract x, y in one pass
            coords = [
                (p.pose.position.x, p.pose.position.y) for p in path.poses
            ]
            xs, ys = np.array(coords).T

            # Vectorized gradient computation using np.diff
            dx = np.diff(xs, append=xs[-1])
            dy = np.diff(ys, append=ys[-1])
            thetas = np.arctan2(dy, dx)

            # For the last point, use direction from second-to-last point
            if n_poses > 1:
                thetas[-1] = thetas[-2]

        # Vectorized array construction for better performance
        ones = np.ones(n_poses)
        # Shape: (4, n_poses)
        initial_point_array = np.vstack([xs, ys, thetas, ones])

        # Convert to list of column vectors for planner API compatibility
        initial_point_list = [
            initial_point_array[:, i:i + 1] for i in range(n_poses)
        ]

        with self._state_lock:
            if (self.neupan_planner.initial_path is None
                    or self.refresh_initial_path):
                self.neupan_planner.set_initial_path(initial_point_list)

    def goal_callback(self, goal: PoseStamped) -> None:
        """Update goal and regenerate initial path.

        Args:
            goal: Goal pose message

        """
        # Extract goal from message (no lock needed)
        x = goal.pose.position.x
        y = goal.pose.position.y
        theta = quat_to_yaw(goal.pose.orientation)

        new_goal = np.array([[x], [y], [theta]])

        self.get_logger().info(
            f"New goal set - x: {x:.2f}m, y: {y:.2f}m, "
            f"theta: {theta:.2f}rad"
        )

        # Check if robot state is ready
        if self.robot_state is None:
            self.get_logger().warn(
                "Goal received but robot state not yet available. "
                "Path planning will start once robot state is received."
            )
            self.goal = new_goal
            return

        # Lock only when accessing shared state and modifying planner
        with self._state_lock:
            self.goal = new_goal

            self.get_logger().debug(
                f"Current state: {self.robot_state.tolist()}"
            )
            self.get_logger().debug(f"Target goal: {self.goal.tolist()}")

            self.neupan_planner.update_initial_path_from_goal(
                self.robot_state, self.goal
            )
            self.neupan_planner.reset()

    def generate_path_msg(self, path_list: List[npt.NDArray]) -> Path:
        """Generate ROS Path message from list of poses.

        Args:
            path_list: List of pose arrays (3, 1) or (4, 1)
                       containing [x, y, theta, ...]

        Returns:
            Path message with poses

        """
        path = Path()
        path.header.frame_id = self.map_frame
        path.header.stamp = self.get_clock().now().to_msg()

        if len(path_list) == 0:
            return path

        # Vectorized approach: normalize all points and stack into matrix
        normalized_points = []
        for point in path_list:
            point_arr = np.array(point)
            if point_arr.ndim == 1:
                point_arr = point_arr.reshape(-1, 1)
            # Extract only first 3 elements (x, y, theta) to ensure consistent dimensions
            point_arr = point_arr[:3, :]
            normalized_points.append(point_arr)

        # Stack all points horizontally -> shape: (3, n_poses)
        points_matrix = np.hstack(normalized_points)

        # Vectorized extraction (single op instead of 3 list comps)
        xs = points_matrix[0, :].tolist()
        ys = points_matrix[1, :].tolist()
        yaws = points_matrix[2, :].tolist()

        # Create path message
        for x, y, yaw in zip(xs, ys, yaws):
            ps = PoseStamped()
            ps.header.frame_id = self.map_frame
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.orientation = yaw_to_quat(yaw)
            path.poses.append(ps)

        return path

    def generate_twist_msg(
            self, vel: Optional[npt.NDArray], stop: bool, arrive: bool
    ) -> Twist:
        """Generate ROS Twist message from velocity command.

        Args:
            vel: Velocity array (2, 1)
                 containing [linear_speed, angular_speed], or None
            stop: Whether the robot should stop (collision risk)
            arrive: Whether the robot has arrived at goal

        Returns:
            Twist message (zero velocity if stopped/arrived or vel is None)

        """
        if vel is None:
            self._cmd_filt_v = 0.0
            self._cmd_filt_w = 0.0
            return Twist()

        speed = float(vel[0, 0])
        steer = float(vel[1, 0])

        if stop or arrive:
            self._cmd_filt_v = 0.0
            self._cmd_filt_w = 0.0
            return Twist()

        speed, steer = self._smooth_cmd(speed, steer)
        action = Twist()
        action.linear.x = speed
        action.angular.z = steer
        return action


def main(args=None):
    """Main entry point for NeuPAN node.

    Args:
        args: Command-line arguments (optional)

    """
    rclpy.init(args=args)

    neupan_node = None
    executor = None
    try:
        neupan_node = NeupanCore()

        # Use MultiThreadedExecutor with 2 threads for concurrent execution
        # Thread 1: run() timer (control loop at configurable frequency)
        # Thread 2: scan/path/goal callbacks (sensor and planning updates)
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(neupan_node)

        viz_status = (
            'enabled' if neupan_node.enable_visualization else 'disabled'
        )
        neupan_node.get_logger().info(
            f"NeuPAN node started - "
            f"Control: {neupan_node.control_frequency}Hz, Threads: 2, "
            f"Visualization: {viz_status}"
        )
        executor.spin()

    except KeyboardInterrupt:
        if neupan_node:
            neupan_node.get_logger().info(
                "NeuPAN node shutting down due to "
                "KeyboardInterrupt (Ctrl+C)."
            )
        pass
    except Exception as e:
        if neupan_node:
            neupan_node.get_logger().error(
                f'Unhandled exception: {e}\n{traceback.format_exc()}'
            )
        raise
    finally:
        if executor:
            executor.shutdown()
        if neupan_node:
            neupan_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
