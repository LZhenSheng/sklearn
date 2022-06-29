from sklearn import datasets
from sklearn.naive_bayes import BernoulliNB

iris = datasets.load_iris()
clf = BernoulliNB()
clf = clf.fit(iris.data, iris.target)
y_pred=clf.predict(iris.data)