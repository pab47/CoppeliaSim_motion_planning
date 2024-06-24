#This code finds optimal path in given environment and start and end goal
import heapq
import numpy as np
import matplotlib.pyplot as plt

# Grid environment
Grid_environment = [
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
]

# Start and goal positions
start = (0, 0)
goal = (4, 8)

# Grid size
grid_size = 0.5  # in meters

# Convert grid coordinates to world coordinates
def grid_to_world(node):
    x = -2.25 + node[1] * grid_size
    y = 2.25 - node[0] * grid_size
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
    if x < 9 and Grid_environment[x + 1][y] == 0:
        neighbors.append((x + 1, y))
    if y > 0 and Grid_environment[x][y - 1] == 0:
        neighbors.append((x, y - 1))
    if y < 9 and Grid_environment[x][y + 1] == 0:
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

# Visualize the environment with the A* path in blue
initial_image = np.zeros((10, 10, 3), dtype=np.uint8)
path_image = np.zeros((10, 10, 3), dtype=np.uint8)  # Initialize path_image

for i in range(10):
    for j in range(10):
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
    i = int((2.25 - y) / grid_size)
    j = int((x + 2.25) / grid_size)
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