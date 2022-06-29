import numpy as np
from sklearn.neighbors import KDTree
rng = np.random.RandomState(0)
x = rng.random_sample((10,3))
tree = KDTree(x,leaf_size=2)
dist,ind = tree.query(x[:1],k=3)
print(ind)