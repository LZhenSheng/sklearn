import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

x = [[2,0],[1,1],[2,3]]
y = [0,0,1]

clf = SVC(kernel='linear')
clf.fit(x,y)
print(clf.predict([[2,2]]))
