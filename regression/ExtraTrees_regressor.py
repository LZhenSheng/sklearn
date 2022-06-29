from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor

x,y = load_diabetes(return_X_y=True)
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)

etr = ExtraTreesRegressor(n_estimators=100,random_state=0).fit(x_train,y_train)
print(etr.score(x_test,y_test))