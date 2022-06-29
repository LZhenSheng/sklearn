import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#VotingClassifier方法是一次使用多种分类模型进行预测，将多数预测结果作为最终结果
x,y = datasets.make_moons(n_samples=500,noise=0.3,random_state=42)

plt.scatter(x[y==0,0],x[y==0,1])
plt.scatter(x[y==1,0],x[y==1,1])
plt.show()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

voting_hard = VotingClassifier(estimators=[
    ('log_clf', LogisticRegression()),
    ('svm_clf', SVC()),
    ('dt_clf', DecisionTreeClassifier(random_state=10)),], voting='hard')

voting_soft = VotingClassifier(estimators=[
    ('log_clf', LogisticRegression()),
    ('svm_clf', SVC(probability=True)),
    ('dt_clf', DecisionTreeClassifier(random_state=10)),
], voting='soft')

voting_hard.fit(x_train,y_train)
print(voting_hard.score(x_test,y_test))

voting_soft.fit(x_train,y_train)
print(voting_soft.score(x_test,y_test))