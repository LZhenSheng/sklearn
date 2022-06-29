from sklearn import svm
from numpy import *

x = array([[0],[1],[2],[3]])
y = array([0,1,2,3])

clf = svm.NuSVC()
clf.fit(x,y)
print(clf.predict([[4]]))