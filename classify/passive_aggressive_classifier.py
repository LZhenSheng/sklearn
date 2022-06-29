from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

x,y = make_classification(n_features=4,random_state=0)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

clf = PassiveAggressiveClassifier(max_iter=1000,random_state=0,tol=1e-3)
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))