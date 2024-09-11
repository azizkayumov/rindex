import matplotlib.pyplot as plt
import numpy as np

# Read the data
data = np.loadtxt("sparse.csv", delimiter=",", skiprows=1, dtype=float)

# Read the edges
edges = np.loadtxt("knn.csv", delimiter=",", dtype=int)

# Plot the edges
for edge in edges:
    point1 = data[edge[0]]
    point2 = data[edge[1]]
    plt.plot([point1[0], point2[0]], [point1[1], point2[1]], c='salmon', linewidth=0.5)

# Plot the data
plt.scatter(data[:, 0], data[:, 1], c='darkred', marker='o', s=1, zorder=2)

# Remove axis labels
plt.xticks([])
plt.yticks([])

# Remove white space
plt.gca().set_axis_off()
plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
plt.margins(0,0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.gca().set_aspect('equal', adjustable='box')

# Save the plot
plt.tight_layout()
plt.savefig("knn.png", dpi=300, transparent=True)