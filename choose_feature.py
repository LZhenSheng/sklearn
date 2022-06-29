from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest,chi2

digits = load_digits()
data = digits.data
target = digits.target
print(data.shape)
data_new = SelectKBest(chi2,k=20).fit_transform(data,target)
print(data_new.shape)

from sklearn.feature_selection import VarianceThreshold

x = [[0,0,1],[0,1,0],[1,0,0],[0,1,1],[0,1,0],[0,1,1]]
vt = VarianceThreshold(threshold=(0.8*(1-0.8)))
x_new = vt.fit_transform(x)
print(x)
print(x_new)

from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

iris = load_iris()
x,y = iris.data,iris.target

lsvc = LinearSVC(C=0.01,penalty='l1',dual=False).fit(x,y)
model = SelectFromModel(lsvc,prefit=True)
x_new = model.transform(x)

print(x.shape)
print(x_new.shape)

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.datasets import load_iris

iris = load_iris()
x,y = iris.data,iris.target

svc = SVC(kernel='linear')
rfecv = RFECV(estimator=svc,step=1,cv=StratifiedKFold(2),scoring='accuracy',verbose=1,n_jobs=1).fit(x,y)
x_rfe = rfecv.transform(x)
print(x_rfe.shape)

clf = SVC(gamma="auto", C=0.8)
scores = (cross_val_score(clf, x_rfe, y, cv=5))
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))