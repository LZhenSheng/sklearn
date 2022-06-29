from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

iris = datasets.load_iris()
clf = GaussianNB()
clf = clf.fit(iris.data,iris.target)

y_pre = clf.predict(iris.data)