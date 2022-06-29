import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

x = np.array([[-1,-1],[-2,-1],[1,1],[2,1]])
y = np.array([1,1,2,2])

clf = make_pipeline(StandardScaler(),SGDClassifier(max_iter=1000,tol=1e-3))
clf.fit(x,y)
print(clf.score(x,y))
print(clf.predict([[-0.8,-1]]))