#Spline fit for At start position.
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# Input path coordinates
path = np.array([
    [-1.75, 2.25],
    [-1.25, 2.25],
    [-1.25, 1.75],
    [-1.25, 1.25],
    [-1.25, 0.75],
    [-1.25, 0.25],
    [-1.25, -0.25],
    [-1.25, -0.75],
    [-1.25, -1.25],
    [-1.25, -1.75],
    [-0.75, -1.75],
    [-0.25, -1.75],
    [-0.25, -1.25],
    [-0.25, -0.75],
    [-0.25, -0.25],
    [0.25, -0.25],
    [0.75, -0.25],
    [1.25, -0.25],
    [1.75, -0.25],
    [1.75, 0.25] 
])

# Number of points between successive input path coordinates
num_points_between = 2

# Extract x and y coordinates from the input path
x_input = path[:, 0]
y_input = path[:, 1]

# Fit a spline to the input path
tck, u = splprep([x_input, y_input], s=0.1)
u_new = np.linspace(u.min(), u.max(), (len(path)-1)*num_points_between + 2)
spline_fit_curve = splev(u_new, tck)

# Output coordinates of the fitted spline with commas
output_coordinates = np.column_stack((spline_fit_curve[0], spline_fit_curve[1]))

# Print the output coordinates with commas and enclosing square brackets
print("Output Coordinates of Fitted Spline (Limited Points):")
print("[")
for i, coord in enumerate(output_coordinates):
    if i == len(output_coordinates) - 1:
        print(f"  [ {coord[0]:.6f}, {coord[1]:.6f} ]")
    else:
        print(f"  [ {coord[0]:.6f}, {coord[1]:.6f} ],")
print("]")

# Plot the input path, the fitted spline, and the output coordinates
plt.figure(figsize=(8, 6))
plt.plot(x_input, y_input, 'ro', label='Input Path')
plt.plot(spline_fit_curve[0], spline_fit_curve[1], 'b-', label='Spline Fit')
plt.scatter(output_coordinates[:, 0], output_coordinates[:, 1], c='g', marker='x', label='Output Coordinates')
plt.title('Spline Fitting on Path Coordinates')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()