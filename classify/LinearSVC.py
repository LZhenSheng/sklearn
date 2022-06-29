import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
x = iris.data
y = iris.target

plt.scatter(x[y==0, 0], x[y==0, 1], color='red')
plt.scatter(x[y==1, 0], x[y==1, 1], color='blue')
plt.show()

svc = LinearSVC(C=10**9)
svc.fit(x, y)
print(svc.score(x,y))