from sklearn.ensemble import BaggingRegressor
from sklearn.datasets import make_regression
from sklearn.svm import SVR

x,y = make_regression(n_samples=100,n_features=4,n_informative=2,n_targets=1,random_state=0,shuffle=False)
br = BaggingRegressor(base_estimator=SVR(),n_estimators=10,random_state=0).fit(x,y)
br.predict([[0,0,0,0]])