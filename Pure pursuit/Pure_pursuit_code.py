#This is final pure pursuit code.
import sim
import math
import time
import sys
import matplotlib.pyplot as plt

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
# Get robot handles
error_code, robot_handle = sim.simxGetObjectHandle(clientID, '/PioneerP3DX', sim.simx_opmode_oneshot_wait)

# Define path coordinates
path = [
  [ -1.738707, 2.262010 ],
  [ -1.455001, 2.296807 ],
  [ -1.291989, 2.216651 ],
  [ -1.220469, 2.048378 ],
  [ -1.211244, 1.818822 ],
  [ -1.235112, 1.554817 ],
  [ -1.262876, 1.283198 ],
  [ -1.269486, 1.027288 ],
  [ -1.256744, 0.787686 ],
  [ -1.237130, 0.555954 ],
  [ -1.223156, 0.323629 ],
  [ -1.225932, 0.083074 ],
  [ -1.243857, -0.165856 ],
  [ -1.268540, -0.419305 ],
  [ -1.291529, -0.673378 ],
  [ -1.304371, -0.924183 ],
  [ -1.298613, -1.167825 ],
  [ -1.265305, -1.399691 ],
  [ -1.187425, -1.603458 ],
  [ -1.043129, -1.753701 ],
  [ -0.846049, -1.833413 ],
  [ -0.642493, -1.834052 ],
  [ -0.463860, -1.756530 ],
  [ -0.324211, -1.611550 ],
  [ -0.236758, -1.410299 ],
  [ -0.214088, -1.164227 ],
  [ -0.239234, -0.897397 ],
  [ -0.253823, -0.651456 ],
  [ -0.218008, -0.454354 ],
  [ -0.127161, -0.309603 ],
  [ 0.020121, -0.218476 ],
  [ 0.225200, -0.182209 ],
  [ 0.482032, -0.194935 ],
  [ 0.768530, -0.235407 ],
  [ 1.060509, -0.280365 ],
  [ 1.333783, -0.306547 ],
  [ 1.564167, -0.290694 ],
  [ 1.727474, -0.209546 ],
  [ 1.799520, -0.039843 ],
  [ 1.756118, 0.241676 ]
]

# Lists to store input and output coordinates for plotting
input_x = [point[0] for point in path]
input_y = [point[1] for point in path]
output_x = []  # Store output x coordinates
output_y = []  # Store output y coordinates

# Pure pursuit control parameters
lookahead_distance = 0.15  # Adjust as needed
linear_velocity = 1.5    # Adjust as needed
goal_radius = 0.1        # Adjust as needed

# Function to get robot pose
def get_robot_pose():
    error_code, robot_position = sim.simxGetObjectPosition(clientID, robot_handle, -1, sim.simx_opmode_streaming)
    error_code, robot_orientation = sim.simxGetObjectOrientation(clientID, robot_handle, -1, sim.simx_opmode_streaming)
    return robot_position, robot_orientation[2]  # Extract yaw angle

# Pure pursuit implementation
def pure_pursuit():
    # Add a flag to track if the final goal has been reached
    if hasattr(pure_pursuit, 'final_goal_reached') and pure_pursuit.final_goal_reached:
        return True  # Return True when the final goal is reached

    robot_position, robot_yaw = get_robot_pose()

    # Find nearest waypoint on path
    nearest_index = 0
    nearest_distance = float('inf')
    for i, waypoint in enumerate(path):
        distance = math.hypot(waypoint[0] - robot_position[0], waypoint[1] - robot_position[1])
        if distance < nearest_distance:
            nearest_index = i
            nearest_distance = distance

    # Print robot coordinates
    print(f"Robot Position: {robot_position}, Yaw Angle: {robot_yaw}")

    # Check if reached goal
    if nearest_distance <= goal_radius:
        # Update goal to the next waypoint
        goal_index = min(nearest_index + 1, len(path) - 1)
        goal_point = path[goal_index]

        # Check if the last waypoint is reached
        if goal_index == len(path) - 1:
            # Stop motors only if the final goal hasn't been reached yet
            if not hasattr(pure_pursuit, 'final_goal_reached') or not pure_pursuit.final_goal_reached:
                sim.simxSetJointTargetVelocity(clientID, left_motor_handle, 0, sim.simx_opmode_oneshot)
                sim.simxSetJointTargetVelocity(clientID, right_motor_handle, 0, sim.simx_opmode_oneshot)
                print("Reached the final goal!")
                pure_pursuit.final_goal_reached = True  # Set the flag to indicate the final goal reached
                return True  # Return True when the final goal is reached

    else:
        goal_point = path[nearest_index]

    # Calculate look-ahead point
    look_ahead_point = path[min(nearest_index + 1, len(path) - 1)]

    # Calculate steering angle
    dx = look_ahead_point[0] - robot_position[0]
    dy = look_ahead_point[1] - robot_position[1]
    angle_to_goal = math.atan2(dy, dx)
    angle_diff = angle_to_goal - robot_yaw

    # Adjust angle_diff to be in the range [-pi, pi]
    if angle_diff > math.pi:
        angle_diff -= 2 * math.pi
    elif angle_diff < -math.pi:
        angle_diff += 2 * math.pi

    # Set motor velocities
    angular_velocity = angle_diff * linear_velocity / lookahead_distance
    left_velocity = linear_velocity - angular_velocity * 0.5 * 0.3  # Adjust wheel separation if needed
    right_velocity = linear_velocity + angular_velocity * 0.5 * 0.3
    sim.simxSetJointTargetVelocity(clientID, left_motor_handle, left_velocity, sim.simx_opmode_oneshot)
    sim.simxSetJointTargetVelocity(clientID, right_motor_handle, right_velocity, sim.simx_opmode_oneshot)
       
    # Append current robot position for plotting
    output_x.append(robot_position[0])
    output_y.append(robot_position[1])

    return False  # Return False if the final goal is not reached yet

# Main loop
while True:
    goal_reached = pure_pursuit()
    time.sleep(0.6)  # Sleep to simulate real-time motion

    # Check if the final goal is reached
    if goal_reached:
        break  # Break the loop when the final goal is reached

# Plotting of the path
# Plot the input path
plt.plot(input_x, input_y, label='Input Path', marker='o')

# Plot the output path
plt.plot(output_x, output_y, label='Output Path', marker='x')

# Mark the start and end points
plt.scatter(input_x[0], input_y[0], color='green', marker='s', label='Start Point')
plt.scatter(input_x[-1], input_y[-1], color='red', marker='s', label='End Point')

# Mark the final robot position
plt.scatter(output_x[-1], output_y[-1], color='blue', marker='D', label='Final Robot Position')

# Set labels and title
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Input and Output Paths')
plt.legend()

# Show the plot
plt.show()