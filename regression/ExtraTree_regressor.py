from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import numpy as np

boston = load_boston()
x = boston.data
y = boston.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

etr = ExtraTreeRegressor()
etr.fit(x_train,y_train)

yetr_pred = etr.predict(x_test)

print(etr.score(x_test,y_test))
print(r2_score(y_test,yetr_pred))