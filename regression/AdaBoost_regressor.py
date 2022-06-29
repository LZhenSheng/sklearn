from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import make_regression

x,y = make_regression(n_features=4,n_informative=2,random_state=0,shuffle=False)
regr = AdaBoostRegressor(random_state=0,n_estimators=100)
regr.fit(x,y)
regr.predict([[0,0,0,0]])