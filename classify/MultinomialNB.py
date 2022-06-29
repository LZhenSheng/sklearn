from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB

iris = datasets.load_iris()
clf = MultinomialNB()
clf = clf.fit(iris.data, iris.target)
y_pred=clf.predict(iris.data)