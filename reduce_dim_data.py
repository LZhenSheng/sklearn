from sklearn.decomposition import PCA
import numpy as np
x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca1 = PCA(n_components=2)
pca2 = PCA(n_components='mle')
pca1.fit(x)
pca2.fit(x)i
x_new1 = pca1.transform(x)
x_new2 = pca2.transform(x)
print(x_new1.shape)
print(x_new2.shape)

import numpy as np
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import math

# kernelPCA适用于对数据进行非线性降维
x = []
y = []
N = 500

for i in range(N):
    deg = np.random.randint(0, 360)
    if np.random.randint(0, 2) % 2 == 0:
        x.append([6 * math.sin(deg), 6 * math.cos(deg)])
        y.append(1)
    else:
        x.append([15 * math.sin(deg), 15 * math.cos(deg)])
        y.append(0)

y = np.array(y)
x = np.array(x)

kpca = KernelPCA(kernel='rbf', n_components=14)
x_kpca = kpca.fit_transform(x)
print(x_kpca.shape)

from sklearn.datasets import load_digits
from sklearn.decomposition import IncrementalPCA
from scipy import sparse

X, _ = load_digits(return_X_y=True)

# 增量主成分分析：适用于大数据
transform = IncrementalPCA(n_components=7, batch_size=200)
transform.partial_fit(X[:100, :])

x_sparse = sparse.csr_matrix(X)
x_transformed = transform.fit_transform(x_sparse)
x_transformed.shape

import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.decomposition import MiniBatchSparsePCA

x, _ = make_friedman1(n_samples=200, n_features=30, random_state=0)
transformer = MiniBatchSparsePCA(n_components=5, batch_size=50, random_state=0)
transformer.fit(x)
x_transformed = transformer.transform(x)
print(x_transformed.shape)

from sklearn.datasets import load_digits
from sklearn.decomposition import FactorAnalysis

x, _ = load_digits(return_X_y=True)
transformer = FactorAnalysis(n_components=7, random_state=0)
x_transformed = transformer.fit_transform(x)
print(x_transformed.shape)