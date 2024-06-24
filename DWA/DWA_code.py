#DWA 
import sim
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

# Connect to CoppeliaSim
sim.simxFinish(-1)
clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)

if clientID != -1:
    print('Connected Successfully.')
else:
    print('Connection Failed.')
    exit()

# Get robot handles
error_code, left_motor_handle = sim.simxGetObjectHandle(clientID, '/PioneerP3DX/leftMotor', sim.simx_opmode_oneshot_wait)
error_code, right_motor_handle = sim.simxGetObjectHandle(clientID, '/PioneerP3DX/rightMotor', sim.simx_opmode_oneshot_wait)
# Get robot handle
error_code, robot_handle = sim.simxGetObjectHandle(clientID, '/PioneerP3DX', sim.simx_opmode_oneshot_wait)

show_animation = True

class RobotType(Enum):
    circle = 0
    rectangle = 1

class Config:
    def __init__(self):
        self.max_speed = 0.5  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.01  # [m/s]
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s]
        self.predict_time = 1.0  # [s]
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 0.8 #1.0
        self.robot_stuck_flag_cons = 0.001
        self.robot_type = RobotType.circle
        self.robot_radius = 0.35/2  # Adjust based on robot size
        self.ob = np.array([[1, 0],
                    [2, -1],
                    [2, -1.2],
                    [2.0, -1.4],
                    [2.0, -1.6],
                    [2.0, -1.8],
                    [2.2, -1.8],
                    [2.4, -1.8],
                    [2.6, -1.8],
                    #[8.0, 10.0],
                    #[9.0, 11.0],
                    #[12.0, 13.0],
                    #[12.0, 12.0],
                    #[15.0, 15.0],
                    #[13.0, 13.0]
                    ])
        #self.ob = []  # No obstacles initially

config = Config()

# Function to get robot pose
def get_robot_pose():
    error_code, robot_position = sim.simxGetObjectPosition(clientID, robot_handle, -1, sim.simx_opmode_buffer)
    error_code, robot_orientation = sim.simxGetObjectOrientation(clientID, robot_handle, -1, sim.simx_opmode_buffer)
    return robot_position, robot_orientation[2]  # Extract yaw angle

def motion(x, u, dt):
    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]
    return x

def calc_dynamic_window(x, config):
    Vs = [config.min_speed, config.max_speed, -config.max_yaw_rate, config.max_yaw_rate]
    Vd = [x[3] - config.max_accel * config.dt, x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt, x[4] + config.max_delta_yaw_rate * config.dt]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]), max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
    return dw

def predict_trajectory(x_init, v, y, config):
    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt
    return trajectory

def calc_control_and_trajectory(x, dw, config, goal, ob):
    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):
            trajectory = predict_trajectory(x_init, v, y, config)
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)
            final_cost = to_goal_cost + speed_cost + ob_cost
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory
                if abs(best_u[0]) < config.robot_stuck_flag_cons and abs(x[3]) < config.robot_stuck_flag_cons:
                    best_u[1] = -config.max_delta_yaw_rate
    return best_u, best_trajectory

def calc_obstacle_cost(trajectory, ob, config):
    ox = ob[:, 0]
    oy = ob[:, 1]
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    r = np.hypot(dx, dy)
    if config.robot_type == RobotType.rectangle:
        yaw = trajectory[:, 2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])
        local_ob = ob[:, None] - trajectory[:, 0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        local_ob = np.array([local_ob @ x for x in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        upper_check = local_ob[:, 0] <= config.robot_length / 2
        right_check = local_ob[:, 1] <= config.robot_width / 2
        bottom_check = local_ob[:, 0] >= -config.robot_length / 2
        left_check = local_ob[:, 1] >= -config.robot_width / 2
        if (np.logical_and(np.logical_and(upper_check, right_check), np.logical_and(bottom_check, left_check))).any():
            return float("Inf")
    elif config.robot_type == RobotType.circle:
        if np.array(r <= config.robot_radius).any():
            return float("Inf")
    min_r = np.min(r)
    return 1.0 / min_r  # OK

def calc_to_goal_cost(trajectory, goal):
    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))
    return cost

def main(gx=3.5, gy= -1.5, robot_type=RobotType.circle):
    print("Dynamic Window Approach motion planning start!!")
    x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])  # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    goal = np.array([gx, gy])  # goal position [x(m), y(m)]
    config.robot_type = robot_type
    trajectory = np.array(x)
    ob = config.ob
    while True:
        dw = calc_dynamic_window(x, config)
        u, predicted_trajectory = calc_control_and_trajectory(x, dw, config, goal, ob)
        x = motion(x, u, config.dt)  # simulate robot
        trajectory = np.vstack((trajectory, x))  # store state history
        # Apply control commands to the robot
        # Example: Set motor velocities based on control commands
        wheel_base = 0.6  # Wheel base of the robot

        # Calculate left and right wheel velocities
        left_velocity = x[3]*2 - x[4] * wheel_base / 2
        right_velocity = x[3]*2 + x[4] * wheel_base / 2
        
        sim.simxSetJointTargetVelocity(clientID, left_motor_handle, left_velocity, sim.simx_opmode_oneshot)
        sim.simxSetJointTargetVelocity(clientID, right_motor_handle, right_velocity, sim.simx_opmode_oneshot)
        if show_animation:
            plt.cla()
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
            plt.plot(x[0], x[1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1], "ok")
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)
        # check reaching goal
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        #if dist_to_goal <= config.robot_radius:
        if dist_to_goal <= 0.1:
            print("Goal!!")
            break
    print("Done")
    if show_animation:
        plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
        plt.pause(0.0001)
        plt.show()

if __name__ == '__main__':
    main()