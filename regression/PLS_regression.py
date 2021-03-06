import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split

boston = datasets.load_boston()

x = boston.data
y = boston.target

x_df = pd.DataFrame(x,columns=boston.feature_names)
y_df = pd.DataFrame(y)

pls = PLSRegression(n_components=2)

x_train,x_test,y_train,y_test = train_test_split(x_df,y_df,test_size=0.3,random_state=1)

pls.fit(x_train,y_train)
print(pls.predict(x_test))