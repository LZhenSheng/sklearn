from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

#加载iris数据集
iris=datasets.load_iris()
X=iris.data
Y=iris.target

#生成K折交叉验证数据
kfold=KFold(n_splits=9)

#决策树及交叉验证
cart=DecisionTreeClassifier(criterion='gini',max_depth=2)
cart=cart.fit(X,Y)
result=cross_val_score(cart,X,Y,cv=kfold)
print('CART数结果:',result.mean())

#装袋法及交叉验证
model=BaggingClassifier(base_estimator=cart,n_estimators=100)
result=cross_val_score(model,X,Y,cv=kfold)
print('装袋法提升后的结果:',result.mean())