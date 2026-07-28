#!/usr/bin/env python3
"""
DC-NeuPAN ROS2 节点：NeuPAN + 观测延迟补偿（真机版）

订阅:  /scan (sensor_msgs/LaserScan), /odom (nav_msgs/Odometry)
发布:  /cmd_vel (geometry_msgs/Twist)

延迟补偿逻辑（与仿真v4一致，但延迟用真实时间戳计算）:
  1. 自身状态补偿: 取scan时刻附近的odom位姿 → 用cmd历史前向推演到
     "动作生效时刻" = now + NeuPAN平均求解耗时
  2. 点云解算: 用scan时刻的位姿把LaserScan转世界点（保证点云参考系正确），
     规划时传入预测后的自身状态
  3. compensate:=false 时退化为原始NeuPAN（A/B对比的对照组）

用法:
  # 补偿开（实验组）
  python3 dc_neupan_node.py --ros-args -p compensate:=true \
      -p planner_yaml:=/home/pi/NeuPAN/example/burger/planner.yaml \
      -p goal_x:=4.0 -p goal_y:=0.0

  # 补偿关（对照组 = 原始NeuPAN）
  python3 dc_neupan_node.py --ros-args -p compensate:=false ...

  # 纯测量模式（不发cmd_vel，只统计真实延迟分布）
  python3 dc_neupan_node.py --ros-args -p measure_only:=true ...
"""
import math
import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

import sys
sys.path.insert(0, '/home/pi/NeuPAN')   # <- 改成OrangePi上NeuPAN的实际路径
from neupan import neupan


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class DCNeuPANNode(Node):
    def __init__(self):
        super().__init__('dc_neupan_node')

        # ---- 参数 ----
        self.declare_parameter('planner_yaml', '/home/pi/NeuPAN/example/burger/planner.yaml')
        self.declare_parameter('compensate', True)
        self.declare_parameter('measure_only', False)
        self.declare_parameter('goal_x', 4.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('goal_tol', 0.3)
        self.declare_parameter('control_rate', 10.0)     # 控制循环频率 Hz
        self.declare_parameter('max_lin', 0.22)           # Burger物理上限
        self.declare_parameter('max_ang', 2.84)

        self.compensate = self.get_parameter('compensate').value
        self.measure_only = self.get_parameter('measure_only').value
        self.goal = np.array([self.get_parameter('goal_x').value,
                              self.get_parameter('goal_y').value])
        self.goal_tol = self.get_parameter('goal_tol').value
        self.max_lin = self.get_parameter('max_lin').value
        self.max_ang = self.get_parameter('max_ang').value
        rate = self.get_parameter('control_rate').value

        # ---- NeuPAN ----
        yaml_path = self.get_parameter('planner_yaml').value
        self.planner = neupan.init_from_yaml(yaml_path)
        self.get_logger().info(f'NeuPAN loaded: {yaml_path}, compensate={self.compensate}')

        # ---- 缓冲区 ----
        self.latest_scan = None                     # (stamp_sec, LaserScan)
        self.odom_hist = deque(maxlen=200)          # (stamp_sec, x, y, yaw)
        self.cmd_hist = deque(maxlen=300)           # (stamp_sec, v, w)
        self.solve_time_ema = 0.3                   # NeuPAN求解耗时EMA初值(s)

        # ---- 统计 ----
        self.stat_scan_age = deque(maxlen=500)
        self.stat_solve = deque(maxlen=500)
        self.arrived = False

        # ---- ROS接口 ----
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, qos)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 20)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(1.0 / rate, self.control_loop)
        self.create_timer(5.0, self.report_stats)

    # ================= 回调 =================
    def scan_cb(self, msg: LaserScan):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.latest_scan = (stamp, msg)

    def odom_cb(self, msg: Odometry):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        yaw = yaw_from_quat(msg.pose.pose.orientation)
        self.odom_hist.append((stamp, p.x, p.y, yaw))

    # ================= 工具 =================
    def now_sec(self):
        t = self.get_clock().now().to_msg()
        return t.sec + t.nanosec * 1e-9

    def pose_at(self, t_query):
        """odom历史里取最接近t_query的位姿"""
        if not self.odom_hist:
            return None
        best = min(self.odom_hist, key=lambda o: abs(o[0] - t_query))
        return np.array([[best[1]], [best[2]], [best[3]]])

    def last_cmd(self):
        return (self.cmd_hist[-1][1], self.cmd_hist[-1][2]) if self.cmd_hist else (0.0, 0.0)

    def rollforward(self, pose_3x1, t_from, t_to):
        """用cmd历史把位姿从t_from推演到t_to（自身状态补偿核心）"""
        state = pose_3x1.copy().flatten()
        cmds = [c for c in self.cmd_hist if c[0] >= t_from]
        if not cmds:
            v, w = self.last_cmd()
            cmds = [(t_from, v, w)]
        # 分段积分：每条cmd生效到下一条cmd（或t_to）
        for k, (tc, v, w) in enumerate(cmds):
            t_next = cmds[k + 1][0] if k + 1 < len(cmds) else t_to
            seg = max(0.0, min(t_next, t_to) - max(tc, t_from))
            state[0] += v * math.cos(state[2]) * seg
            state[1] += v * math.sin(state[2]) * seg
            state[2] += w * seg
        return state.reshape(3, 1)

    # ================= 主循环 =================
    def control_loop(self):
        if self.latest_scan is None or not self.odom_hist or self.arrived:
            return

        t_now = self.now_sec()
        scan_stamp, scan_msg = self.latest_scan
        scan_age = t_now - scan_stamp
        self.stat_scan_age.append(scan_age)

        # ---- 到达判定 ----
        cur = self.odom_hist[-1]
        if np.linalg.norm(np.array([cur[1], cur[2]]) - self.goal) < self.goal_tol:
            self.cmd_pub.publish(Twist())  # 停车
            if not self.arrived:
                self.get_logger().info('=== GOAL REACHED ===')
            self.arrived = True
            return

        # ---- 构造planner输入 ----
        pose_scan = self.pose_at(scan_stamp)   # scan时刻的位姿（点云参考系）
        if pose_scan is None:
            return

        if self.compensate:
            # 预测"动作生效时刻"的自身状态: t_apply = now + 预计求解耗时
            t_apply = t_now + self.solve_time_ema
            state_in = self.rollforward(pose_scan, scan_stamp, t_apply)
        else:
            state_in = pose_scan   # 原始NeuPAN

        scan_dict = {
            'ranges': list(scan_msg.ranges),
            'angle_min': scan_msg.angle_min,
            'angle_max': scan_msg.angle_max,
            'range_min': scan_msg.range_min,
            'range_max': scan_msg.range_max,
        }

        # 点云用scan时刻位姿解算（参考系正确），规划状态用补偿后的
        try:
            points = self.planner.scan_to_point(pose_scan, scan_dict)
        except Exception as e:
            self.get_logger().warn(f'scan_to_point failed: {e}')
            return

        # ---- NeuPAN求解（计时） ----
        t0 = time.monotonic()
        try:
            action, info = self.planner(state_in, points, None)
        except Exception as e:
            self.get_logger().warn(f'planner failed: {e}')
            return
        solve = time.monotonic() - t0
        self.stat_solve.append(solve)
        self.solve_time_ema = 0.9 * self.solve_time_ema + 0.1 * solve

        if info.get('arrive', False):
            self.cmd_pub.publish(Twist())
            self.arrived = True
            self.get_logger().info('=== NeuPAN arrive flag ===')
            return

        v = float(np.clip(action[0, 0], -self.max_lin, self.max_lin))
        w = float(np.clip(action[1, 0], -self.max_ang, self.max_ang))

        if self.measure_only:
            return  # 只统计，不驱动

        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self.cmd_pub.publish(msg)
        self.cmd_hist.append((self.now_sec(), v, w))

    def report_stats(self):
        if self.stat_scan_age:
            sa = np.array(self.stat_scan_age)
            sv = np.array(self.stat_solve) if self.stat_solve else np.array([0.0])
            self.get_logger().info(
                f'[delay] scan_age: mean={sa.mean()*1000:.0f}ms max={sa.max()*1000:.0f}ms | '
                f'solve: mean={sv.mean()*1000:.0f}ms max={sv.max()*1000:.0f}ms | '
                f'total obs-delay ~= {(sa.mean()+sv.mean())*1000:.0f}ms')


def main():
    rclpy.init()
    node = DCNeuPANNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.cmd_pub.publish(Twist())  # 退出时停车
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
