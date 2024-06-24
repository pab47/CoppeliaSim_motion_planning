#Author: Saurabh Ramesh Bharane, University of Illinois at Chicago, MIE Department

#Astar
#This code finds optimal path in given environment to reach from start to end goal
import heapq
import numpy as np
import matplotlib.pyplot as plt

# Grid environment (Modify as needed, 0 = Free space, 1 = Occupied space)
Grid_environment = [
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
]

# Start and goal positions (Modify as needed)
start = (1, 1)
goal = (25, 25)

# Grid size
grid_size = 0.5  # in meters

# Convert grid coordinates to world coordinates (Modify when Environment size changes)
def grid_to_world(node):
    x = -7.25 + node[1] * grid_size
    y = 7.25 - node[0] * grid_size
    return x, y

# Heuristic function (Manhattan distance)
def heuristic(node, goal):
    x1, y1 = grid_to_world(node)
    x2, y2 = grid_to_world(goal)
    return abs(x1 - x2) + abs(y1 - y2)

# Define the neighbors function
def get_neighbors(node):
    x, y = node
    neighbors = []
    if x > 0 and Grid_environment[x - 1][y] == 0:
        neighbors.append((x - 1, y))
    if x < 29 and Grid_environment[x + 1][y] == 0:
        neighbors.append((x + 1, y))
    if y > 0 and Grid_environment[x][y - 1] == 0:
        neighbors.append((x, y - 1))
    if y < 29 and Grid_environment[x][y + 1] == 0:
        neighbors.append((x, y + 1))
    return neighbors

# Define the A* algorithm
def astar(grid, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while open_set:
        current_cost, current_node = heapq.heappop(open_set)

        if current_node == goal:
            break

        for neighbor in get_neighbors(current_node):
            new_cost = cost_so_far[current_node] + 1  # Assuming a cost of 1 for each step
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                heapq.heappush(open_set, (priority, neighbor))
                came_from[neighbor] = current_node

    # Reconstruct the path
    path_grid = []
    path_coordinates = []
    current = goal
    while current is not None:
        path_grid.append(current)
        path_coordinates.append(grid_to_world(current))
        current = came_from[current]
    path_grid.reverse()
    path_coordinates.reverse()

    return path_grid, path_coordinates

# Find the optimal path
optimal_path_grid, optimal_path_coordinates = astar(Grid_environment, start, goal)

print("Optimal Path (Grid):", optimal_path_grid)
print("Optimal Path (World Coordinates):", optimal_path_coordinates)



#Spline fit for smoothening of A star path.
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# Input path coordinates
astar_path = np.array(optimal_path_coordinates)

# Number of points between successive input path coordinates
num_points_between = 2 # Change for variation in curvature

# Extract x and y coordinates from the input path
x_input = astar_path[:, 0]
y_input = astar_path[:, 1]

# Fit a spline to the input path
tck, u = splprep([x_input, y_input], s=0.1) # Change "s"for variation in curvature
u_new = np.linspace(u.min(), u.max(), (len(astar_path)-1)*num_points_between + 2)
spline_fit_curve = splev(u_new, tck)

# Output coordinates of the fitted spline with commas
output_coordinates = np.column_stack((spline_fit_curve[0], spline_fit_curve[1]))




#This is pure pursuit code for robot trajectory generation and P-3DX movement.
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
path = (output_coordinates)

# Lists to store input and output coordinates for plotting
input_x = [point[0] for point in path]
input_y = [point[1] for point in path]
output_x = []  # Store output x coordinates
output_y = []  # Store output y coordinates

# Pure pursuit control parameters
lookahead_distance = 0.15  # Adjust as needed
linear_velocity = 2.0    # Adjust as needed
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



# Visualize the environment with the A* path in blue
initial_image = np.zeros((30, 30, 3), dtype=np.uint8)
path_image = np.zeros((30, 30, 3), dtype=np.uint8)  # Initialize path_image

for i in range(30):
    for j in range(30):
        if Grid_environment[i][j] == 0:  # Open space (green)
            initial_image[i, j] = [0, 255, 0]
            path_image[i, j] = [0, 255, 0]
        elif Grid_environment[i][j] == 1:  # Obstacle (red)
            initial_image[i, j] = [255, 0, 0]
            path_image[i, j] = [255, 0, 0]

# Mark start as orange and goal as yellow
initial_image[start[0], start[1]] = [255, 165, 0]  # Orange
initial_image[goal[0], goal[1]] = [255, 255, 0]   # Yellow

path_image[start[0], start[1]] = [255, 165, 0]  # Orange
path_image[goal[0], goal[1]] = [255, 255, 0]   # Yellow

# Mark the A* path as blue
for x, y in optimal_path_coordinates:
    i = int((7.25 - y) / grid_size)
    j = int((x + 7.25) / grid_size)
    path_image[i, j] = [0, 0, 255]  # Blue

# Plot the initial environment
plt.imshow(initial_image)
plt.title("Initial Environment")
plt.show()

# Plot the path image
plt.imshow(path_image)
plt.title("A* Path Planning (Grid)")
plt.show()

# Save the images
plt.imsave("initial_environment.png", initial_image)
plt.imsave("astar_path_grid.png", path_image)