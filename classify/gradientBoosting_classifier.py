import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

#make_blobs:sklearn中自带的取类数据生成器随机生成测试样本，make_blobs方法中n_samples表示生成的随机数样本数量，n_features表示每个样本的特征数量，centers表示类别数量，random_state表示随机种子
x,y=make_blobs(n_samples=1000,n_features=6,centers=50,random_state=0)
plt.scatter(x[:,0],x[:,1],c=y)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x,y)

# 模型训练，使用GBDT算法
gbr = GradientBoostingClassifier(n_estimators=3000, max_depth=2, min_samples_split=2, learning_rate=0.1)
gbr.fit(x_train, y_train.ravel())

y_gbr = gbr.predict(x_train)
y_gbr1 = gbr.predict(x_test)
acc_train = gbr.score(x_train, y_train)
acc_test = gbr.score(x_test, y_test)
print(acc_train)
print(acc_test)