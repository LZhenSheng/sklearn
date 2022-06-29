import numpy as np
from sklearn import preprocessing

# 标准化：将数据转换为均值为0，方差为1的数据，即标注正态分布的数据
x = np.array([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
x_scale = preprocessing.scale(x)
print(x_scale.mean(axis=0), x_scale.std(axis=0))

std_scale = preprocessing.StandardScaler().fit(x)
x_std = std_scale.transform(x)
print(x_std.mean(axis=0), x_std.std(axis=0))

# 将数据缩放至给定范围（0-1）
mm_scale = preprocessing.MinMaxScaler()
x_mm = mm_scale.fit_transform(x)
print(x_mm.mean(axis=0), x_mm.std(axis=0))

# 将数据缩放至给定范围（-1-1）,适用于稀疏数据
mb_scale = preprocessing.MaxAbsScaler()
x_mb = mb_scale.fit_transform(x)
print(x_mb.mean(axis=0), x_mb.std(axis=0))

# 适用于带有异常值的数据
rob_scale = preprocessing.RobustScaler()
x_rob = rob_scale.fit_transform(x)
print(x_rob.mean(axis=0), x_rob.std(axis=0))

# 正则化
nor_scale = preprocessing.Normalizer()
x_nor = nor_scale.fit_transform(x)
print(x_nor.mean(axis=0), x_nor.std(axis=0))

# 特征二值化：将数值型特征转换位布尔型的值
bin_scale = preprocessing.Binarizer()
x_bin = bin_scale.fit_transform(x)
print(x_bin)

# 将分类特征或数据标签转换位独热编码
ohe = preprocessing.OneHotEncoder()
x1 = ([[0, 0, 3], [1, 1, 0], [1, 0, 2]])
x_ohe = ohe.fit(x1).transform([[0, 1, 3]])
print(x_ohe)

import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.arange(6).reshape(3, 2)
poly = PolynomialFeatures(2)
x_poly = poly.fit_transform(x)
print(x)
print(x_poly)

import numpy as np
from sklearn.preprocessing import FunctionTransformer

# 自定义的特征转换函数
transformer = FunctionTransformer(np.log1p)

x = np.array([[0, 1], [2, 3]])
x_trans = transformer.transform(x)
print(x_trans)

import numpy as np
import sklearn.preprocessing

x = np.array([[-3, 5, 15], [0, 6, 14], [6, 3, 11]])
kbd = preprocessing.KBinsDiscretizer(n_bins=[3, 2, 2], encode='ordinal').fit(x)
x_kbd = kbd.transform(x)
print(x_kbd)

from sklearn.preprocessing import MultiLabelBinarizer

# 多标签二值化
mlb = MultiLabelBinarizer()
x_mlb = mlb.fit_transform([(1, 2), (3, 4), (5,)])
print(x_mlb)
