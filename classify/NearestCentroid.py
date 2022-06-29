from sklearn.neighbors import NearestCentroid
import numpy as np

x = np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
y = np.array([1,1,1,2,2,2])

clf = NearestCentroid()
clf.fit(x,y)
print(clf.predict([[-0.8,-1]]))