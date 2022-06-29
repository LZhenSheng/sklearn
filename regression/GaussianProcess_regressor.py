from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct,WhiteKernel

x,y = make_friedman2(n_samples=500,noise=0,random_state=0)

kernel = DotProduct()+WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(x,y)
print(gpr.score(x,y))