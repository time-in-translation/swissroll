import string

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import datasets

# Next line to silence pyflakes. This import is needed.
Axes3D

n_points = 1000
X, color = datasets.make_swiss_roll(n_points, random_state=0)

# Create figure
fig = plt.figure(figsize=(15, 8))

# Add 3d scatter plot
ax = fig.add_subplot(221, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.zaxis.set_major_formatter(NullFormatter())
ax.view_init(4, -72)

ax = fig.add_subplot(222)

# Add points and lines
xs = np.array([2.5, 2.75, 3.0, 3.0, 2.75, 2.5, 2.25, 2.0, 1.75, 1.65, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.35, 3.25, 3.0, 2.75, 2.5, 2.25])
ys = np.array([2.5, 2.60, 3.0, 3.5, 3.85, 4.0, 3.85, 3.5, 3.00, 2.38, 1.75, 1.0, 0.90, 1.0, 1.15, 1.5, 2.00, 3.00, 4.00, 4.5, 4.85, 5.0, 5.10])
ax.plot(xs, ys, 'bo-')  # Geodesic distance
ax.plot((xs[6], xs[len(xs) - 1]), (ys[6], ys[len(xs) - 1]), 'ro-')  # Euclidean distance
ax.legend(['Geodesic dist.', 'Euclidean dist.'], loc='upper left')

# Add annotations
labels = np.array(list(string.ascii_uppercase[:len(xs)])[::-1])
for x, y, label in zip(xs, ys, labels):
    plt.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4))

# Remove axis
ax.axis('off')

plt.show()
