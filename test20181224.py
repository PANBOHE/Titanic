# -*- coding: utf-8 -*-
# @File  : test20181224.py
# @Author: Panbo
# @Date  : 2018/12/24
# @Desc  :
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#数据目录
train_data = pd.read_csv(r"D://studyPythonMl//kaggle_competition//Titannic//source_data//train.csv")

#train_data.head()  查看表信息
#train_data.info()  查看数据信息

#观察获救比例
'''
fig =  plt.figure()
train_data.Survived.value_counts().plot(kind = 'bar')
plt.ylabel('人数')
plt.title('获救情况')

#解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.show()
'''
#数据预处理

#补全缺失值Age、Cabin、Emarked

#利用随机森林预测缺失的age值，补全

#定义补全函数
from sklearn.ensemble import RandomForestRegressor


def set_miss_age(df):
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]  # 只拿出数值型的特征预测Age
    know_age = age_df[age_df.Age.notnull()].as_matrix()  # 已知age值的特征作为训练集
    unknow_age = age_df[age_df.Age.isnull()].as_matrix()  # 未知的作为测试集预测年龄
    X = know_age[:, 1:]
    y = know_age[:, 0]
    test_age = unknow_age[:, 1:]
    model = RandomForestRegressor(random_state=2018, n_estimators=2000, n_jobs=-1)
    model.fit(X, y)
    pre_age = model.predict(test_age)
    df.loc[(df.Age.isnull()), 'Age'] = pre_age
    return df, model

train_data,model = set_miss_age(train_data)

#Cabin缺失值用No填充、非缺失值用Yes填充
def set_cabin(df):
    df.loc[(df.Cabin.notnull()),'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()),'Cabin'] = 'No'
    return df

train_data = set_cabin(train_data)

#Embarked缺失值3个，用较多的S填充  众数
train_data.Embarked = train_data.Embarked.fillna('S')
#利用pd.get_dummies把Cabin、Embarked、Sex、Pclass做成one-hot数据（特征因子化）
dum_cabin = pd.get_dummies(train_data.Cabin, prefix='Cabin')
dum_embark = pd.get_dummies(train_data.Embarked, prefix='Embarked')
dum_sex = pd.get_dummies(train_data.Sex, prefix='Sex')
dum_pclass = pd.get_dummies(train_data.Pclass, prefix='Pclass')

df = pd.concat([train_data, dum_cabin, dum_embark, dum_sex, dum_pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

# Name、Ticket特征丢弃
#年龄，fare 归一化，有利于梯度下降
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1, 1))

#处理test

data_test = pd.read_csv(r"D://studyPythonMl//kaggle_competition//Titannic//source_data//test.csv")
tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()

X = null_age[:, 1:]
predictedAges = model.predict(X)
data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges

data_test = set_cabin(data_test)
dum_cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
dum_embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
dum_sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
dum_pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')

df_test = pd.concat([data_test, dum_cabin, dum_embarked, dum_sex, dum_pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1, 1))
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1, 1))


#test设计完，用训练集
train_df = df[
    ['Survived', 'Age_scaled', 'SibSp', 'Parch', 'Fare_scaled', 'Cabin_Yes', 'Cabin_No', 'Embarked_C', 'Embarked_S',
     'Embarked_Q', 'Sex_female', 'Sex_male', 'Pclass_1', 'Pclass_2', 'Pclass_3']]
train_np = train_df.as_matrix()

X = train_np[:, 1:]
y = train_np[:, 0]


#用lr

from sklearn import linear_model
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import StratifiedKFold

LR = linear_model.LogisticRegression(penalty='l1', tol=1e-6)

if __name__ == '__main__':
    param_grid = dict(C=[0.01, 0.1, 0.2, 0.3, 0.5, 1.0])  # 正则化参数
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2018)  # 10折交叉验证
    GS = GridSearchCV(LR, param_grid, scoring='accuracy', n_jobs=-1, cv=list(kfold.split(X, y)))
    GS_result = GS.fit(X, y)

    print('Best: %f using %s' % (GS_result.best_score_, GS_result.best_params_))

    test_cols = train_df.columns.tolist()[1:len(train_df.columns)]
    test = df_test[test_cols]
    predictions = GS_result.predict(test)
    result = pd.DataFrame(
        {'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
    result.to_csv(r'predictions_LR.csv', index=False)
'''
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import StratifiedKFold

if __name__ == '__main__':
    RF = RF()
    param_grid = dict(max_features=[3, 4, 5, 6], n_estimators=[100, 200, 300], min_samples_leaf=[3, 4, 5, 6])
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2018)
    GS = GridSearchCV(RF, param_grid, scoring='accuracy', n_jobs=-1, cv=list(kfold.split(X, y)))
    GS_result = GS.fit(X, y)
    print('Best: %f using %s' % (GS_result.best_score_, GS_result.best_params_))

    test_cols = train_df.columns.tolist()[1:len(train_df.columns)]
    test = df_test[test_cols]
    predictions = GS_result.predict(test)
    result = pd.DataFrame(
        {'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
    result.to_csv(r'predictions_RF.csv', index=False)
'''



