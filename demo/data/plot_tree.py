import matplotlib.pyplot as plt


# Read the data
lines = open("tree.csv").readlines()

points = {}
bubbles = {}
for row in lines:
    s = row.split(",")
    id = int(s[0])
    height = int(s[1])
    if height == 0:
        x, y = float(s[-2]), float(s[-1])
        points[id] = (x, y)
    elif height == 1:
        radius = float(s[-3])
        x, y = float(s[-2]), float(s[-1])
        children = s[3].split("+")
        bubbles[id] = (x, y, radius, children)

fig, ax = plt.subplots()
colors = ['r', 'g', 'b', 'y', 'c', 'm']
for id, (x, y, radius, children) in bubbles.items():
    color = colors[id % len(colors)]
    circle = plt.Circle((x, y), radius, color='b', alpha=0.25)
    ax.add_artist(circle)
    for child in children:
        child = int(child)
        x, y = points[child][0], points[child][1]
        plt.scatter(x, y, color='b', s=2, marker='+')

# remove coordinate axis names
plt.xticks([])
plt.yticks([])
plt.gca().set_aspect('equal')

title = "N = " + str(len(points)) + ", L = " + str(len(bubbles))
plt.title(title)

# set axis limits
plt.savefig("tree.png", dpi=300)