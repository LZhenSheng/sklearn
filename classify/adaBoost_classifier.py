from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

#加载数据
dataset_all=datasets.load_breast_cancer()
X=dataset_all.data
Y=dataset_all.target

#初始化基本随机数生成器
kfold=KFold(n_splits=10)

#决策树及交叉验证
dtree=DecisionTreeClassifier(criterion='gini',max_depth=3)

#提升法及交叉验证
model=AdaBoostClassifier(base_estimator=dtree,n_estimators=100)
result=cross_val_score(model,X,Y,cv=kfold)
print("提升法改进结果:",result.mean())