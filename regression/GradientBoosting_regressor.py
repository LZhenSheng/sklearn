import numpy as np
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.datasets import make_regression

x, y = make_regression(1000, 2, noise=10)

gbr = GBR()
gbr.fit(x, y)
gbr_preds = gbr.predict(x)