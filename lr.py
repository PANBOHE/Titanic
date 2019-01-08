# -*- coding: utf-8 -*-
# @File  : lr.py
# @Author: Panbo
# @Date  : 2018/12/19
# @Desc  :

import pandas as pd  # pandas数据分析数据处理库


titanic = pd.read_csv("D://studyPythonMl//kaggle_competition//Titannic//source_data//train.csv")

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
# median 返回给定数值的中值，中值是在一组数值中居于中间的数值
# fillna对缺失值进行填充
'''
mean 平均
std 标准差
'''

# 把性别转换为0,1
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

# 把上船地点转换为0,1,2
titanic["Embarked"] = titanic["Embarked"].fillna('S')
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

'''线性回归'''
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


#print(alg.coef_)
#print(alg.intercept_)


xishu =alg.coef_
jieju =alg.intercept_
LR=  LinearRegression()
LR.intercept_=jieju
LR.coef_=xishu

testdate = pd.read_csv("D://studyPythonMl//kaggle_competition//Titannic//source_data//test.csv")

testdate["Age"] = testdate["Age"].fillna(testdate["Age"].median())
# median 返回给定数值的中值，中值是在一组数值中居于中间的数值
# fillna对缺失值进行填充
'''
mean 平均
std 标准差
'''

# 把性别转换为0,1
testdate.loc[testdate["Sex"] == "male", "Sex"] = 0
testdate.loc[testdate["Sex"] == "female", "Sex"] = 1

# 把上船地点转换为0,1,2
testdate["Embarked"] = testdate["Embarked"].fillna('S')
testdate.loc[testdate["Embarked"] == "S", "Embarked"] = 0
testdate.loc[testdate["Embarked"] == "C", "Embarked"] = 1
testdate.loc[testdate["Embarked"] == "Q", "Embarked"] = 2


predictions2 = []
for train, test in kf:
    # print(train)  # 297-890 | 0-296 594-890 | 0-593
    # print(test)  # 0-296 | 297-593 | 594-890

    X_test = testdate[predictors].iloc[test, :]
    y_test = LR.predict(X_test)
    predictions2.append(y_test)

print("预测成功")
'''
import numpy as np

predictions = np.concatenate(predictions)
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0

accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print("The accuracy of the LR is :",accuracy)

result = pd.DataFrame({'PassengerId':titanic['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv(r'predictions_stack.csv', index=False)
'''