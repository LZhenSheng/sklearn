from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

x,y=make_blobs(n_samples=1000,n_features=6,centers=50,random_state=0)
plt.scatter(x[:,0],x[:,1],c=y)
plt.show()

#构造随机森林模型
clf=RandomForestClassifier(n_estimators=10,max_depth=None,min_samples_split=2,random_state=0)
scores=cross_val_score(clf,x,y)
print('RandomForestClassifier result:',scores.mean())