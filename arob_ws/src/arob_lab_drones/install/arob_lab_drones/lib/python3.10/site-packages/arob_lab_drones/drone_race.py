import rclpy
import os
from rclpy.node import Node
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Header, ColorRGBA
from builtin_interfaces.msg import Duration
import tf_transformations
import copy
from time import sleep
from geometry_msgs.msg import PoseStamped, TwistStamped
import math
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from std_srvs.srv import SetBool
from as2_python_api.drone_interface import DroneInterface
from as2_msgs.srv import SetControlMode
from arob_lab_drones.trajectory_generator import TrajectoryGenerator  # adjust path if needed
import argparse
from types import SimpleNamespace
import time



class DroneRaceNode(Node):
    def __init__(self):
        super().__init__('drone_race_node')

        self.uav = None
        self.set_control_mode_client = None
        self.trajectory = None  # Will hold the TrajectoryGenerator instance
        
        # Create subscriber
        # QoS compatible con publicadores "best effort"
        qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT
        )
        self.pose_suscriber = self.create_subscription(PoseStamped, '/drone0/self_localization/pose',
            self.drone_pose,   # callback
            qos_profile        # QoS queue size
        )
        # Create publishers
        self.pose_command_publisher = self.create_publisher(PoseStamped, '/drone0/motion_reference/pose', 10)
        self.vel_command_publisher = self.create_publisher(TwistStamped, '/drone0/motion_reference/twist', 10)

        # For LAB 2
        self.current_gate = PoseStamped()
        self.current_gate_id = 0
        self.current_pose = PoseStamped()

        # For LAB 3
        self.goal_idx = 0

        # Velocity suscriber for position PID
        self.twist_suscriber = self.create_subscription(TwistStamped, '/drone0/self_localization/twist',
            self.drone_twist,   # callback
            qos_profile        # QoS queue size
        )
        self.current_twist = TwistStamped()
        self.pos_integral = 0
        self.pos_cmd = 0

        # Publishers for markers
        marker_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,  # <-- latched in ROS 2
        )
        self.pub_gate_markers = self.create_publisher(MarkerArray, 'gate_markers', marker_qos)
        self.pub_traj_markers = self.create_publisher(MarkerArray, 'trajectory_markers', marker_qos)
        
        self.id_marker = 0
        self.gates = []

        ################################################################################
        ##### MPPI params #####
        self.mppi_dt = 0.05            # should match your timer period in velocity mode
        self.mppi_T  = 10#25              # horizon steps (25*0.05=1.25s)
        self.mppi_K  = 64 #512             # rollouts
        self.mppi_lambda = 1.0         # temperature

        # Control limits [vx, vy, vz, yaw_rate]
        #self.u_min = np.array([-2.0, -2.0, -1.0, -2.0], dtype=float)
        #self.u_max = np.array([ 2.0,  2.0,  1.0,  2.0], dtype=float)

        self.u_min = np.array([-2.5, -2.5, -1.5, -2.0], dtype=float)  # Slightly more aggressive
        self.u_max = np.array([ 2.5,  2.5,  1.5,  2.0], dtype=float)

        # Exploration noise std
        # self.noise_std = np.array([0.6, 0.6, 0.4, 0.7], dtype=float)
        # Increased exploration
        self.noise_std = np.array([0.4, 0.4, 0.25, 0.4], dtype=float)

        # Cost weights
        self.w_pos = 25.0#12.0
        self.w_vel = 3.0
        self.w_yaw = 0.0#0.5
        self.w_u   = 0.01#0.05
        self.w_du  = 0.05#0.2

        # Nominal control sequence U[t] = [vx, vy, vz, yaw_rate]
        self.U = np.zeros((self.mppi_T, 4), dtype=float)

        # For time alignment with your generated trajectory
        self.t_start_wall = None  # set on first control tick
        ################################################################################

    # MPPI 
    ################################################################################

    # We compute the wrapped difference: This keeps yaw error “shortest way around the circle”. near the ±pi boundary
    def wrap_pi(self, a):
        a = float(a)
        return (a + np.pi) % (2*np.pi) - np.pi

    # We need yaw in the state, the simulator can predict yaw over the horizon.
    # our reference may include yaw (face the next gate / face velocity direction)
    def yaw_from_quat(self, q):
        # q type quternion
        quat = [q.x, q.y, q.z, q.w]
        _, _, yaw = tf_transformations.euler_from_quaternion(quat)
        return float(yaw)

    # MPPI starts from current state
    def get_state(self):
        # Need pose + twist
        if self.current_pose is None or self.current_twist is None:
            return None

        p = self.current_pose.pose.position
        v = self.current_twist.twist.linear
        yaw = self.yaw_from_quat(self.current_pose.pose.orientation)

        x = np.array([p.x, p.y, p.z, v.x, v.y, v.z, yaw], dtype=float)
        return x     
    
    # Cost uses reference Position_ref(t), Velocity_ref(t), yaw_ref(t) along the horizon
    def ref_at_time(self, t):
        #
        #Returns reference position, velocity, yaw at time t (seconds since trajectory start).
        #
        # pick the reference point on your precomputed trajectory at time t:
        # Clamp into trajectory time range
        # Clamping guarantees: reference lookup always returns a valid point.
        t = float(t)
        #t = float(np.clip(t, float(self.traj_times[0]), float(self.traj_times[-1])))

        # Nearest neighbor (simple, fast). You can upgrade to interpolation later.
        # the sample whose timestamp is closest to the requested time t
        #idx = int(np.argmin(np.abs(self.traj_times - t))) # gives an array of differences to t, makes all differences positive, returns the index of the smallest difference


        idx = int(round((t - self.traj_times[0]) / self.mppi_dt))
        idx = max(0, min(idx, len(self.traj_times)-1))

        pref = self.traj_positions[idx].copy()
        vref = self.traj_velocities[idx].copy()

        # Yaw reference: face direction of velocity (if moving), else keep current yaw=0
        speed = np.linalg.norm(vref)
        if speed > 1e-3:
            yaw_ref = math.atan2(vref[1], vref[0])
        else:
            yaw_ref = 0.0

        return pref, vref, yaw_ref
    
    """
    def ref_at_time(self, t):
        
        #Returns reference position, velocity, yaw at time t (seconds since trajectory start).
        
        t = float(t)
        t_clamped = float(np.clip(t, float(self.traj_times[0]), float(self.traj_times[-1])))
        
        # ADD THIS WARNING
        if abs(t - t_clamped) > 0.01:  # If clamping happened
            if not hasattr(self, '_clamp_warned'):
                self._clamp_warned = True
                self.get_logger().warn(
                    f'Time clamping: t={t:.2f} -> {t_clamped:.2f}, '
                    f'traj range=[{self.traj_times[0]:.2f}, {self.traj_times[-1]:.2f}]'
                )
        
        idx = int(np.argmin(np.abs(self.traj_times - t_clamped)))
        
        pref = self.traj_positions[idx].copy()
        vref = self.traj_velocities[idx].copy()
        
        # ADD THIS LOG (only occasionally)
        if not hasattr(self, '_ref_log_counter'):
            self._ref_log_counter = 0
        self._ref_log_counter += 1
        if self._ref_log_counter % 100 == 0:  # Log every 5 seconds at 50ms
            self.get_logger().info(
                f'ref_at_time: t={t:.2f}, idx={idx}/{len(self.traj_times)}, '
                f'pref=[{pref[0]:.3f}, {pref[1]:.3f}, {pref[2]:.3f}]'
            )
        
        speed = np.linalg.norm(vref)
        if speed > 1e-3:
            yaw_ref = math.atan2(vref[1], vref[0])
        else:
            yaw_ref = 0.0
        
        return pref, vref, yaw_ref   
    """
    # Dynamics model for rollouts
    def dynamics_step(self, x, u):
        """
        x = [px,py,pz,vx,vy,vz,yaw]
        u = [vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd]
        """
        dt = self.mppi_dt
        tau = 0.2 #0.25  # velocity tracking aggressiveness

        p = x[0:3]
        v = x[3:6]
        yaw = x[6]

        v_cmd = u[0:3]
        r_cmd = u[3]

        v_next = v + (dt / tau) * (v_cmd - v)
        p_next = p + dt * v_next
        yaw_next = self.wrap_pi(yaw + dt * r_cmd)

        x_next = x.copy()
        x_next[0:3] = p_next
        x_next[3:6] = v_next
        x_next[6] = yaw_next
        return x_next    
    
    # Cost function
    def rollout_cost(self, X: np.ndarray, U: np.ndarray, t0: float) -> float:
        """
        X shape: (T+1, 7), U shape: (T, 4)
        t0: start time on the reference trajectory (seconds)
        """
        J = 0.0
        prev_u = U[0]

        for t in range(self.mppi_T):
            pref, vref, yaw_ref = self.ref_at_time(t0 + t * self.mppi_dt)

            p = X[t+1, 0:3]
            v = X[t+1, 3:6]
            yaw = X[t+1, 6]
            u = U[t]

            #This is the squared Euclidean distance to the reference.
            pos_err = p - pref
            vel_err = v - vref
            yaw_err = self.wrap_pi(yaw - yaw_ref)

            #This cost comes directly from optimal control theory.
            # This is standard MPC / LQR / MPPI theory:
            J += self.w_pos * float(pos_err @ pos_err)
            J += self.w_vel * float(vel_err @ vel_err)
            J += self.w_yaw * float(yaw_err * yaw_err)

            J += self.w_u * float(u @ u)
            du = u - prev_u
            J += self.w_du * float(du @ du)
            prev_u = u

        return J
    
    # MPPI update   
    # https://acdslab.github.io/mppi-generic-website/docs/mppi.html

    def mppi_optimize(self, x0: np.ndarray, t0):
        T = self.mppi_T
        K = self.mppi_K

        noise = np.random.normal(0.0, 1.0, size=(K, T, 4)) * self.noise_std[None, None, :]
        U_samples = self.U[None, :, :] + noise
        U_samples = np.clip(U_samples, self.u_min[None, None, :], self.u_max[None, None, :])

        costs = np.zeros(K, dtype=float)

        for k in range(K):
            X = np.zeros((T + 1, 7), dtype=float)
            X[0] = x0
            for t in range(T):
                X[t+1] = self.dynamics_step(X[t], U_samples[k, t])
            costs[k] = self.rollout_cost(X, U_samples[k], t0)

        Jmin = float(np.min(costs))
        lam = max(1e-6, float(self.mppi_lambda))
        w = np.exp(-(costs - Jmin) / lam)
        w /= (np.sum(w) + 1e-12)

        self.U = np.tensordot(w, U_samples, axes=(0, 0))  # (T,4)
        u0 = self.U[0].copy()

        # Shift for next iteration
        self.U[:-1] = self.U[1:]
        self.U[-1] = self.U[-2]
        return u0
    ################################################################################

    def start_drone(self):
        print('Initializing Aerostack Drone')
        # Drone interface
        self.uav = DroneInterface(
            drone_id='drone0',
            use_sim_time=True,
            verbose=True)
        ##### ARM OFFBOARD #####
        print("Offboard")
        self.uav.offboard()
        print("Arm")
        self.uav.arm()

        ##### TAKE OFF #####
        print("Take Off")
        self.uav.takeoff(height=1.0, speed=1.0)
        sleep(1.0)
        print('Drone ready to fly')

        # Create control mode client
        self.set_control_mode_client = self.create_client(SetControlMode, '/drone0/controller/set_control_mode')

    def set_control_mode(self, yaw_mode=1, control_mode=2, reference_frame=1):
        """Send a SetControlMode service request to the drone controller."""
        # Wait until service is available
        if not self.set_control_mode_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('Service /drone0/controller/set_control_mode not available')
            return False

        # Build the request
        req = SetControlMode.Request()
        req.control_mode.header.stamp.sec = 0
        req.control_mode.header.stamp.nanosec = 0
        req.control_mode.header.frame_id = ''
        req.control_mode.yaw_mode = yaw_mode
        req.control_mode.control_mode = control_mode
        req.control_mode.reference_frame = reference_frame

        # Call service synchronously
        future = self.set_control_mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        # Process result
        if future.result() is not None:
            success = future.result().success
            self.get_logger().info(f"SetControlMode call success={success}")
            return success
        else:
            self.get_logger().error(f"Service call failed: {future.exception()}")
            return False
    
    def read_gates(self, filepath):
        self.gates.clear()
        try:
            with open(filepath, 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        x, y, z, yaw = map(float, parts[0:4])
                        pose = Pose()
                        pose.position.x = x
                        pose.position.y = y
                        pose.position.z = z
                        # Orientation from yaw only, assuming no roll/pitch
                        quat = tf_transformations.quaternion_from_euler(0, 0, yaw)
                        pose.orientation.x = quat[0]
                        pose.orientation.y = quat[1]
                        pose.orientation.z = quat[2]
                        pose.orientation.w = quat[3]
                        self.gates.append(pose)
            self.get_logger().info(f'Read {len(self.gates)} gates from {filepath}')

            # Initialize the first goal
            gate = self.gates[0]
            self.current_gate.header.stamp = self.get_clock().now().to_msg()
            self.current_gate.header.frame_id = 'earth'

            # Pose
            self.current_gate.pose.position.x = gate.position.x
            self.current_gate.pose.position.y = gate.position.y
            self.current_gate.pose.position.z = gate.position.z

            # Orientation
            self.current_gate.pose.orientation.x = gate.orientation.x
            self.current_gate.pose.orientation.y = gate.orientation.y
            self.current_gate.pose.orientation.z = gate.orientation.z
            self.current_gate.pose.orientation.w = gate.orientation.w
            self.current_gate_id = 0
            return True
        except Exception as e:
            self.get_logger().error(f'Error reading gates file: {e}')
            return False
    
    def drone_pose(self, msg: PoseStamped):
        self.current_pose = copy.deepcopy(msg)
        # print('Pose received')
    
    def drone_twist(self, msg: TwistStamped):
        self.current_twist = copy.deepcopy(msg)

    def RPY_to_R_matrix(roll, pitch, yaw):
        """
        Converts the roll, pitch, and yaw angles to a 3×3 rotation matrix (NumPy).
        Similar to Eigen::AngleAxis and Eigen::Quaternion used in C++.
        """
        q = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
        R = tf_transformations.quaternion_matrix(q)[:3, :3]
        return R

    def quat_to_R_matrix(self, q):
        """
        Converts a geometry_msgs.msg.Quaternion to a 3×3 rotation matrix (NumPy).
        Uses tf_transformations for the conversion.
        """
        quat = [q.x, q.y, q.z, q.w]
        R = tf_transformations.quaternion_matrix(quat)[:3, :3]
        return R

    def RPY_to_quat(self, roll, pitch, yaw):
        """
        Converts roll, pitch, and yaw to a geometry_msgs.msg.Quaternion
        """
        q = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
        quat_msg = Quaternion()
        quat_msg.x = q[0]
        quat_msg.y = q[1]
        quat_msg.z = q[2]
        quat_msg.w = q[3]
        return quat_msg
    
    def draw_gate_markers(self):
        marker_array = MarkerArray()
        # print(f'Drawing {len(self.gates)} gates as markers')
        for idx, gate in enumerate(self.gates):

            rotate_gate = self.quat_to_R_matrix(gate.orientation)
            pos_gate = np.array([gate.position.x, gate.position.y, gate.position.z])

            # marker = Marker()
            # marker.header.frame_id = 'earth'
            # # marker.header.stamp = self.get_clock().now().to_msg()
            # marker.header.stamp.sec = 0
            # marker.header.stamp.nanosec = 0
            # marker.ns = 'gates'
            # marker.type = Marker.CUBE
            # marker.action = Marker.ADD
            # marker.scale.x = 0.2
            # marker.scale.y = 0.2
            # marker.scale.z = 0.2
            # marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
            # marker.lifetime = Duration(sec=0)

            # Line Marker for the green lines
            line_marker = Marker()
            line_marker.header.frame_id = "earth"
            # line_marker.header.stamp = self.get_clock().now().to_msg()
            line_marker.header.stamp.sec = 0
            line_marker.header.stamp.nanosec = 0
            line_marker.ns = "line"
            line_marker.id = self.id_marker
            self.id_marker += 1
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.scale.x = 0.05
            line_marker.pose.orientation.w = 1.0
            line_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            line_marker.lifetime = Duration(sec=0)

            gate_size = 0.75
            points = []

            # Gate Corners
            corners = [
                np.array([0.0, gate_size, gate_size]),
                np.array([0.0, -gate_size, gate_size]),
                np.array([0.0, -gate_size, -gate_size]),
                np.array([0.0, gate_size, -gate_size])
            ]

            marker = Marker()
            marker.header.frame_id = 'earth'
            marker.header.stamp.sec = 0
            marker.header.stamp.nanosec = 0
            marker.ns = 'gates'
            marker.lifetime = Duration(sec=0)
            marker.action = Marker.ADD
            for corner in corners:    
                pos = pos_gate + rotate_gate @ corner
                marker.pose.position.x = float(pos[0])
                marker.pose.position.y = float(pos[1])
                marker.pose.position.z = float(pos[2])
                # Modify Corners
                if (corner[1] > 0 ): # If X is in front, the left is on positive values of y
                    marker.type = Marker.SPHERE
                    marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
                    marker.scale.x = 0.25
                    marker.scale.y = 0.25
                    marker.scale.z = 0.25   
                else:
                    marker.type = Marker.CUBE
                    marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
                    marker.scale.x = 0.2
                    marker.scale.y = 0.2
                    marker.scale.z = 0.2

                marker.id = self.id_marker
                self.id_marker += 1
                marker_array.markers.append(copy.deepcopy(marker))  # We add each cube

                # Last point in the line
                point = Point()
                point.x = float(pos[0])
                point.y = float(pos[1])
                point.z = float(pos[2])
                

                points.append(point)

            # Close the square by connecting the last point to the first.
            points.append(points[0])
            line_marker.points = points

            marker_array.markers.append(copy.deepcopy(line_marker))

        self.pub_gate_markers.publish(marker_array)
        # self.get_logger().info('Published gate markers')
    
    def generate_trajectory_example(self):
        """
        Example to illustrate how to compute a trajectory
        """

        traj_gen = TrajectoryGenerator()
        traj_gen.add_waypoint([0, 0, 0], 0) # Add waypoint [0,0,0] at time t=0
        traj_gen.add_waypoint([0, 0, 1.5], 1) # Add waypoint [0,0,0] at time t=1
        traj_gen.add_waypoint([0, 2, 2], 3) # Add waypoint [0,0,0] at time t=3
        traj_gen.add_waypoint([2, 2, 1.5], 4) # Add waypoint [0,0,0] at time t=4
        traj_gen.add_waypoint([2, 0, 1], 5) # Add waypoint [0,0,0] at time t=5
        traj_gen.add_waypoint([0, 0, 1], 8) # Add waypoint [0,0,0] at time t=8

        # Constraints
        traj_gen.add_waypoint_constraint(derivative_order=1, wpt_index=0, value=[0.0, 0.0, 0.0], side="right")   # v(t0)
        traj_gen.add_waypoint_constraint(derivative_order=1, wpt_index=len(traj_gen.times)-1, value=[0,0,0], side="left")   # v(tf)
        traj_gen.add_waypoint_constraint(derivative_order=2, wpt_index=len(traj_gen.times)-1, value=[0,0,0], side="left")   # a(tf) 

        # Compute trajectory
        # opts = SimpleNamespace(trajectory_type="piecewise_polynomial", order=3)
        opts = SimpleNamespace(trajectory_type="minimum_snap", order=7, continuity_order=3, reg=1e-9, normalized=True)
        traj_gen.generate_trajectory(opts)

        self.trajectory = traj_gen

        # Store the trajectory in class variables
        deltat = 0.05
        t0 = float(self.trajectory.times[0])
        tf = float(self.trajectory.times[-1])
        n_steps = int(np.floor((tf - t0) / deltat)) + 1
        ts = t0 + np.arange(n_steps, dtype=float) * deltat
        if ts[-1] < tf - 1e-9:
            ts = np.append(ts, tf)

        positions = np.zeros((ts.size, 3), dtype=float)
        velocities = np.zeros((ts.size, 3), dtype=float)

        for i, t in enumerate(ts):
            positions[i, :]  = traj_gen.evaluate_trajectory(t, derivative_order=0)
            velocities[i, :] = traj_gen.evaluate_trajectory(t, derivative_order=1)
        # Store
        self.traj_times = ts
        self.traj_positions = positions
        self.traj_velocities = velocities

        self.get_logger().info("Trajectory successfully generated.")


    
    def generate_trajectory(self):
        """
        Generate a trajectory through the sequence of gates
        using the TrajectoryGenerator class.
        """
        # Complete the code to create a trajectory that goes through all the gates as fast as possible
        traj_gen = TrajectoryGenerator()
        time = 0
        gain = 0.95 #Time calculator based on a gain
        traj_gen.add_waypoint([0, 0, 1], time) # Add waypoint [0,0,1] at time t=0
        for gate in self.gates:

            ## Assure to go through the gate. This assumes the drone is passing from behind the gate.
            ## We obtain the 0.5 m before the gate
            rotate_gate = self.quat_to_R_matrix(gate.orientation)
            offset_local_prev = np.array([-0.5, 0, 0]) 
            gate_waypoint = np.array([gate.position.x, gate.position.y, gate.position.z])
            gate_prev = gate_waypoint + rotate_gate @ offset_local_prev
            
            distance = np.linalg.norm(gate_prev - traj_gen.waypoints[-1]) # To set a time based on the distance from the prev waypoint
            time += distance * gain #We obtain the time based on the distance and a gain
            traj_gen.add_waypoint(gate_prev, time)

            ## Assure to go through the gate. This assumes the drone is passing from behind the gate.
            ## We obtain the 0.5 m after the gate
            offset_local_post = np.array([0.5, 0, 0])
            gate_post = gate_waypoint + rotate_gate @ offset_local_post
            distance = np.linalg.norm(gate_post - traj_gen.waypoints[-1]) # To set a time based on the distance from the prev waypoint
            time += distance * gain #We obtain the time based on the distance and a gain
            traj_gen.add_waypoint(gate_post, time)

        ## Time calculator based on a mean velocity
        end = np.array([0, 0, 1])
        distance = np.linalg.norm(end - traj_gen.waypoints[-1]) # To set a time based on the distance from the prev waypoint
        time += distance * gain
        traj_gen.add_waypoint(end, time) # Add waypoint [0,0,1] at the end t=0
        print("end",time)

        # Constraints
        traj_gen.add_waypoint_constraint(derivative_order=1, wpt_index=0, value=[0.0, 0.0, 0.0], side="right")   # v(t0)
        traj_gen.add_waypoint_constraint(derivative_order=1, wpt_index=len(traj_gen.times)-1, value=[0,0,0], side="left")   # v(tf)
        traj_gen.add_waypoint_constraint(derivative_order=2, wpt_index=len(traj_gen.times)-1, value=[0,0,0], side="left")   # a(tf) 

        # Compute trajectory
        opts = SimpleNamespace(trajectory_type="minimum_snap", order=7, continuity_order=3, reg=1e-9, normalized=True)
        traj_gen.generate_trajectory(opts)

        self.trajectory = traj_gen

        # Store the trajectory in class variables
        deltat = 0.05
        t0 = float(self.trajectory.times[0])
        tf = float(self.trajectory.times[-1])
        n_steps = int(np.floor((tf - t0) / deltat)) + 1
        ts = t0 + np.arange(n_steps, dtype=float) * deltat
        if ts[-1] < tf - 1e-9:
            ts = np.append(ts, tf)

        positions = np.zeros((ts.size, 3), dtype=float)
        velocities = np.zeros((ts.size, 3), dtype=float)
        accelerations = np.zeros((ts.size, 3), dtype=float)

        for i, t in enumerate(ts):
            positions[i, :]  = traj_gen.evaluate_trajectory(t, derivative_order=0)
            velocities[i, :] = traj_gen.evaluate_trajectory(t, derivative_order=1)
            accelerations[i, :] = traj_gen.evaluate_trajectory(t, derivative_order=2)
        # Store
        self.traj_times = ts
        self.traj_positions = positions
        self.traj_velocities = velocities
        self.traj_accelerations = accelerations

        self.get_logger().info("Trajectory successfully generated.")

    def generate_simple_test(self):
        """Dead simple trajectory for testing"""
        traj_gen = TrajectoryGenerator()
        
        # Just a simple square at z=1.5m, SLOW
        traj_gen.add_waypoint([0, 0, 1.0], 0)
        traj_gen.add_waypoint([0, 0, 1.5], 3)   # Rise slowly
        traj_gen.add_waypoint([1, 0, 1.5], 6)   # Move forward
        traj_gen.add_waypoint([1, 1, 1.5], 9)   # Move right
        traj_gen.add_waypoint([0, 1, 1.5], 12)  # Move back
        traj_gen.add_waypoint([0, 0, 1.5], 15)  # Return
        traj_gen.add_waypoint([0, 0, 1.0], 18)  # Land
        
        # Constraints
        traj_gen.add_waypoint_constraint(derivative_order=1, wpt_index=0, value=[0.0, 0.0, 0.0], side="right")
        traj_gen.add_waypoint_constraint(derivative_order=1, wpt_index=len(traj_gen.times)-1, value=[0,0,0], side="left")
        traj_gen.add_waypoint_constraint(derivative_order=2, wpt_index=len(traj_gen.times)-1, value=[0,0,0], side="left")
        
        opts = SimpleNamespace(trajectory_type="minimum_snap", order=7, continuity_order=3, reg=1e-9, normalized=True)
        traj_gen.generate_trajectory(opts)

        # VERIFY the trajectory was actually generated
        if not hasattr(traj_gen, 'trajectory') or traj_gen.trajectory is None:
            self.get_logger().error("TRAJECTORY GENERATION FAILED!")
            return
        
        self.trajectory = traj_gen
        
        # Store trajectory
        deltat = 0.05
        t0 = float(self.trajectory.times[0])
        tf = float(self.trajectory.times[-1])

        # ADD THESE LOGS:
        #self.get_logger().info(f"Trajectory time range: t0={t0:.2f}s, tf={tf:.2f}s")
        #self.get_logger().info(f"Waypoint times: {self.trajectory.times}")

        n_steps = int(np.floor((tf - t0) / deltat)) + 1
        ts = t0 + np.arange(n_steps, dtype=float) * deltat
        if ts[-1] < tf - 1e-9:
            ts = np.append(ts, tf)
        
        positions = np.zeros((ts.size, 3), dtype=float)
        velocities = np.zeros((ts.size, 3), dtype=float)
        
        for i, t in enumerate(ts):
            positions[i, :]  = traj_gen.evaluate_trajectory(t, derivative_order=0)
            velocities[i, :] = traj_gen.evaluate_trajectory(t, derivative_order=1)
        
        self.traj_times = ts
        self.traj_positions = positions
        self.traj_velocities = velocities
        
        self.get_logger().info(f"Simple test trajectory: {tf}s duration")
        # ADD THESE LOGS:
        #self.get_logger().info(f"Stored trajectory samples: {len(self.traj_times)} points")
        #self.get_logger().info(f"First 5 positions: {self.traj_positions[:5]}")
        #self.get_logger().info(f"Last 5 positions: {self.traj_positions[-5:]}")
        #self.get_logger().info(f"Position at t=10s: {self.traj_positions[int(10/deltat)]}")        

        
    def draw_trajectory_markers(
        self,
        deltat: float = 0.1,
        vel_scale: float = 0.3,
        shaft_diam: float = 0.03,
        head_diam: float = 0.06,
        head_len: float = 0.08,
        max_arrow_len: float | None = None,
    ):
        """
        Visualize the spline trajectory in RViz using a MarkerArray:
        - LINE_STRIP for the full path,
        - SPHERE_LIST for sampled points,
        - ARROW markers for velocity vectors (direction = velocity, length ∝ |v|).
        The arrow length is |v| * vel_scale (optionally clamped by max_arrow_len).
        """
        # TODO: use existing class attributes instead of evaluate the trajectory again 
        
        # Preconditions
        if not hasattr(self, 'trajectory') or self.trajectory is None:
            self.get_logger().warn('No trajectory generated yet. Call generate_trajectory() first.')
            return
        if not hasattr(self.trajectory, 'trajectory') or self.trajectory.trajectory is None:
            self.get_logger().warn('Spline has not been computed. Call compute_spline() on the trajectory.')
            return
        if deltat <= 0.0:
            self.get_logger().warn('deltat must be > 0.')
            return

        # Build time samples [t0, tf] with fixed step; ensure exact inclusion of tf
        t0 = float(self.trajectory.times[0])
        tf = float(self.trajectory.times[-1])
        n_steps = int(np.floor((tf - t0) / deltat)) + 1
        ts = t0 + np.arange(n_steps, dtype=float) * deltat
        if ts[-1] < tf - 1e-9:
            ts = np.append(ts, tf)

        # Evaluate positions and velocities
        pts, vels = [], []
        for t in ts:
            pos = self.trajectory.evaluate_trajectory(t)
            vel = self.trajectory.evaluate_trajectory(t, 1)
            pts.append(Point(x=pos[0], y=pos[1], z=pos[2]))
            vels.append(tuple(vel))

        # 1) Trajectory path (line)
        line_marker = Marker()
        line_marker.header.frame_id = "earth"
        # line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.header.stamp.sec = 0
        line_marker.header.stamp.nanosec = 0
        line_marker.ns = "trajectory_line"
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.pose.orientation.w = 1.0
        line_marker.scale.x = 0.03  # line thickness
        line_marker.color = ColorRGBA(r=0.0, g=0.6, b=1.0, a=1.0)
        line_marker.lifetime = Duration(sec=0)
        line_marker.points = pts

        # 2) Sampled points (spheres)
        dots_marker = Marker()
        dots_marker.header.frame_id = "earth"
        # dots_marker.header.stamp = self.get_clock().now().to_msg()
        dots_marker.header.stamp.sec = 0
        dots_marker.header.stamp.nanosec = 0
        dots_marker.ns = "trajectory_points"
        dots_marker.id = 1
        dots_marker.type = Marker.SPHERE_LIST
        dots_marker.action = Marker.ADD
        dots_marker.pose.orientation.w = 1.0
        dots_marker.scale.x = 0.06
        dots_marker.scale.y = 0.06
        dots_marker.scale.z = 0.06
        dots_marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
        dots_marker.lifetime = Duration(sec=0)
        dots_marker.points = pts

        # 3) Velocity arrows
        # Use ARROW with two points per marker: start (p) and end (p + dir * length).
        # When using 'points', the arrow length is the distance between the two points.
        arrow_markers = []
        for i, (p, v) in enumerate(zip(pts, vels)):
            vx, vy, vz = v
            speed = float(np.sqrt(vx*vx + vy*vy + vz*vz))

            # Skip zero-speed (or near-zero) to avoid degenerate arrows
            if speed < 1e-6:
                continue

            # Compute arrow length from speed and optional clamp
            length = speed * float(vel_scale)
            if max_arrow_len is not None:
                length = min(length, float(max_arrow_len))

            # Direction = normalized velocity
            inv = 1.0 / speed
            dx, dy, dz = vx * inv * length, vy * inv * length, vz * inv * length

            end = Point(x=p.x + dx, y=p.y + dy, z=p.z + dz)

            arrow = Marker()
            arrow.header.frame_id = "earth"
            # arrow.header.stamp = self.get_clock().now().to_msg()
            arrow.header.stamp.sec = 0
            arrow.header.stamp.nanosec = 0
            arrow.ns = "velocity_arrows"
            arrow.id = 100 + i  # avoid collision with other IDs
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose.orientation.w = 1.0  # ignored when using 'points' for ARROW

            # Arrow defined by two points: start and end
            arrow.points = [p, end]

            # Geometry: when using points, scale.x = shaft diameter, scale.y = head diameter, scale.z = head length
            arrow.scale.x = float(shaft_diam)
            arrow.scale.y = float(head_diam)
            arrow.scale.z = float(head_len)

            # Color (magenta); adjust alpha if you want translucency
            arrow.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0)

            arrow.lifetime = Duration(sec=0)
            arrow_markers.append(arrow)

        # Publish everything together
        marray = MarkerArray()
        marray.markers.append(line_marker)
        marray.markers.append(dots_marker)
        marray.markers.extend(arrow_markers)
        self.pub_traj_markers.publish(marray)
    
    def position_timer_callback(self):
        self.get_logger().info('position timer callback')

        if self.current_pose is None:
            return
        # Complete this function to publish position commands to follow the trajectory 
    
    def velocity_timer_callback(self):
        #self.get_logger().info('velocity timer callback')       
        #if int(time.time()) % 1 == 0:
            #self.get_logger().info("velocity timer callback")
        if self.current_pose is None or self.current_twist is None:
            return
        if not hasattr(self, 'traj_times'):
            return  # trajectory not generated

        # Start time reference
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t_start_wall is None:
            self.t_start_wall = now

        # Time along the reference trajectory
        t_ref = now - self.t_start_wall

        x0 = self.get_state()
        if x0 is None:
            return
        t0 = time.perf_counter()
        u0 = self.mppi_optimize(x0, t_ref)
        t1 = time.perf_counter()
        #self.get_logger().info(f"MPPI compute time: {(t1-t0)*1000:.1f} ms")
        # Publish command
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "earth"
        msg.twist.linear.x = float(u0[0])
        msg.twist.linear.y = float(u0[1])
        msg.twist.linear.z = float(u0[2])
        msg.twist.angular.z = float(u0[3])
        self.vel_command_publisher.publish(msg)        
  

def main(args=None):
    rclpy.init(args=args)

    # --- Parse command-line arguments ---
    parser = argparse.ArgumentParser(description="Drone Race Trajectory Script")
    parser.add_argument("--no-drone", action="store_true",
                        help="Run without connecting to the drone (for trajectory testing only).")
    parser.add_argument("--vel-control", action="store_true",
                        help="Use velocity control mode instead of position control.")
    cli_args, _ = parser.parse_known_args()

    node = DroneRaceNode()
    print('Drone Race Node launched')

    # Reading the gates
    #path = '/root/arob_ws/src/arob_lab_drones/data/gates.txt'
    path = '/root/arob_ws/src/arob_lab_drones/data/gates_2.txt'
    filepath = os.path.expanduser(path)
    filepath = os.path.abspath(filepath)
    success = node.read_gates(filepath)
    if success:
        # node.generate_trajectory_example()
        node.generate_trajectory() #Original
        #node.generate_simple_test()
        # Republish markers every 2 seconds so RViz always sees them
        node.create_timer(2.0, node.draw_gate_markers)
        node.create_timer(2.0, node.draw_trajectory_markers)
    else:
        node.get_logger().error('Failed to start node due to gates file error')

    if not cli_args.no_drone:
        node.start_drone()
        if cli_args.vel_control:
            node.get_logger().info('Controlling drone in VELOCITY mode')
            #node.set_control_mode(yaw_mode=2, control_mode=3, reference_frame=2) # Velocity control #Original
            node.set_control_mode(yaw_mode=2, control_mode=3, reference_frame=2) # Velocity control
            sleep(2.0)
            #node.create_timer(0.05, node.velocity_timer_callback) # Original
            node.create_timer(0.15, node.velocity_timer_callback)
        else:
            node.get_logger().info('Controlling drone in POSITION mode')
            node.set_control_mode(yaw_mode=1, control_mode=2, reference_frame=1) # Position control
            sleep(2.0)
            node.create_timer(0.05, node.position_timer_callback)
    else:
        print("Running in --no-drone mode: skipping Aerostack drone interface.")
        
    
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()