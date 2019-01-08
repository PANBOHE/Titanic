# -*- coding: utf-8 -*-
# @File  : test.py
# @Author: Panbo
# @Date  : 2018/12/19
# @Desc  : 泰坦尼克号，灾难问题。
import pandas as pd #数据分析
import numpy as np #科学计算
from sklearn import linear_model
from sklearn.model_selection import StratifiedKFold


#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)

titanic = pd.read_csv("D://studyPythonMl//kaggle_competition//Titannic//source_data//train.csv")
'''
print( titanic.describe()) #std代表方差，Age中存在缺失
#以下操作为对数据进行预处理
#算法大多是矩阵运算，不能存在缺失值，用均值来填充缺失值
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

print( titanic.describe())
'''

#sex是字符串，无法进行计算，将它转成数字，用0代表man，1代表female
print(titanic["Sex"].unique())

titanic.loc[titanic["Sex"]=="male","Sex"] = 0
titanic.loc[titanic["Sex"]=="female","Sex"] = 1


#登船的地点也是字符串，需要变换成数字,并填充缺失值
print (titanic["Embarked"].unique())
titanic["Embarked"] = titanic["Embarked"].fillna('S')
#loc通过索引获取数据
titanic.loc[titanic["Embarked"]=="S","Embarked"] = 0
titanic.loc[titanic["Embarked"]=="C","Embarked"] = 1
titanic.loc[titanic["Embarked"]=="Q","Embarked"] = 2

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
alg = LinearRegression()
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    # print(train)  # 297-890 | 0-296 594-890 | 0-593
    # print(test)  # 0-296 | 297-593 | 594-890
    X_train = titanic[predictors].iloc[train, :]
    y_train = titanic["Survived"].iloc[train]
    alg.fit(X_train, y_train)
    X_test = titanic[predictors].iloc[test, :]
    y_test = alg.predict(X_test)
    predictions.append(y_test)

import numpy as np

predictions = np.concatenate(predictions)
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0

accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print(accuracy)
