from sklearn import datasets
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

clf = QDA()
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))