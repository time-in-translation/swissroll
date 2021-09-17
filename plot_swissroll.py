import string

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import datasets

# Next line to silence pyflakes. This import is needed.
Axes3D

# Generate swiss roll dataset (with a fixed random state)
n_points = 1000
X, color = datasets.make_swiss_roll(n_points, random_state=0)

# Create figure to show swiss roll in 3D
fig = plt.figure(figsize=(15, 8),)
ax = fig.add_subplot(projection='3d')
# Add 3d scatter plot
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.zaxis.set_major_formatter(NullFormatter())
ax.view_init(4, -72)
plt.show()

# Create figure to compare geodesic and euclidean distance
# Add points and lines
xs = np.array([2.5, 2.75, 3.0, 3.0, 2.75, 2.5, 2.25, 2.0, 1.75, 1.65, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.35, 3.25, 3.0, 2.75, 2.5, 2.25])
ys = np.array([2.5, 2.60, 3.0, 3.5, 3.85, 4.0, 3.85, 3.5, 3.00, 2.38, 1.75, 1.0, 0.90, 1.0, 1.15, 1.5, 2.00, 3.00, 4.00, 4.5, 4.85, 5.0, 5.10])
plt.plot(xs, ys, 'bo-')  # Geodesic distance
plt.plot((xs[6], xs[len(xs) - 1]), (ys[6], ys[len(xs) - 1]), 'ro-')  # Euclidean distance
plt.legend(['Geodesic dist.', 'Euclidean dist.'], loc='upper left')
plt.axis('off')

# Add annotations
labels = np.array(list(string.ascii_uppercase[:len(xs)])[::-1])
for x, y, label in zip(xs, ys, labels):
    plt.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4))

plt.savefig('geodesic-euclidean.png')
plt.show()
