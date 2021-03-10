# Adapted from https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html
# Original author: Jake Vanderplas -- <vanderplas@astro.washington.edu>

from collections import OrderedDict
from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold, datasets

# Next line to silence pyflakes. This import is needed.
Axes3D

n_points = 1000
X, color = datasets.make_swiss_roll(n_points, random_state=0)
n_neighbors = 10
n_components = 2
rs = 0  # fix the random state
perplexity = 80

# Create figure
fig = plt.figure(figsize=(15, 8))

# Add 3d scatter plot
ax = fig.add_subplot(241, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.zaxis.set_major_formatter(NullFormatter())
ax.view_init(4, -72)

# Set-up manifold methods
methods = OrderedDict()
methods['MDS'] = manifold.MDS(n_components, n_init=1, random_state=rs)
methods['LLE'] = manifold.LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors, random_state=rs)
methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca', random_state=rs, perplexity=50)

# Plot results
for i, (label, method) in enumerate(methods.items()):
    t0 = time()
    Y = method.fit_transform(X)
    t1 = time()
    print("%s: %.2g sec" % (label, t1 - t0))
    ax = fig.add_subplot(2, 4, 2 + i + (i > 3))
    ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    ax.set_title("%s" % label)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis('tight')

plt.show()
