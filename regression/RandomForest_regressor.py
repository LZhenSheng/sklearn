from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

x,y = make_regression(n_features=4,n_informative=2,random_state=0,shuffle=False)

rfr = RandomForestRegressor(max_depth=2,random_state=0)
rfr.fit(x,y)
print(rfr.predict([[0,0,0,0]]))