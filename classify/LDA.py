from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

iris = datasets.load_iris()
X = iris.data[:-5]
pre_x = iris.data[-5:]
y = iris.target[:-5]
print ('first 10 raw samples:', X[:10])
clf = LDA()
clf.fit(X, y)
X_r = clf.transform(X)
pre_y = clf.predict(pre_x)
#降维结果
print ('first 10 transformed samples:', X_r[:10])
#预测目标分类结果
print ('predict value:', pre_y)