import numpy as np
from sklearn import linear_model,datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(x_train,y_train)

prepro = logreg.score(x_test,y_test)
print(prepro)