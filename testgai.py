# -*- coding: utf-8 -*-
# @File  : testgai.py
# @Author: Panbo
# @Date  : 2018/12/25
# @Desc  : 增加了数据处理
import pandas as pd  # pandas 是Python中数据处理
import numpy as np   #numpy 是Python 开源数值扩展。
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


#数据导入
train = pd.read_csv(r"D://studyPythonMl//kaggle_competition//Titannic//source_data//train.csv", dtype={"Age": np.float64})
test = pd.read_csv(r"D://studyPythonMl//kaggle_competition//Titannic//source_data//test.csv", dtype={"Age": np.float64})

PassengerId = test['PassengerId']
all_data = pd.concat([train, test], ignore_index = True)


#数据可视化和数据分析。
#print(train.info())
#主要是年龄和客舱的缺失比较大。

#1：性别和幸存的关系
import seaborn as sns   # Seaborn  属于Matplotlib的一个高级接口，为我们进行数据可视化分析提供方便。
#%matplotlib inline  这是让图片在notebook 汇总显示数据点。

sns.barplot(x = "Sex", y="Survived", data=train)
#2 船舱等级与是否幸存的关系
sns.barplot(x= "Pclass", y ="Survived", data= train)
#3 年龄和幸存率的关系
facet1 = sns.FacetGrid(train, hue="Survived",aspect=2)
facet1.map(sns.kdeplot,'Age',shade= True)
facet1.set(xlim=(0,train['Age'].max()))
facet1.add_legend()

#4 兄弟姐妹数和幸存率的关系


plt.figure()
ax= sns.barplot(x="SibSp", y ="Survived", data = train, palette = 'Set3')
plt.show()


#5. 父母子女数与幸存关系
plt.figure()
sns.barplot( x = "Parch", y = "Survived", data=train, palette='Set3')
plt.show()
#6. 票价与幸存者的关系
train['Fare'].describe()

plt.figure()
facet2 = sns.FacetGrid(train, hue="Survived", aspect=2)
facet2.map(sns.kdeplot, 'Fare',shade=True)
facet2.set(xlim = (0, 200))
facet2.add_legend()
plt.show()

#船舱类型和幸存者
#1）：分为  有值：1与缺失值：0 进行分析
#2）：缺失值填充字符，  与其他一起进行分析。
all_data['Cabin'] = all_data['Cabin'].fillna('Unknown')
all_data['Deck'] = all_data['Cabin'].str.get(0)

plt.figure()
facet4=sns.barplot(x='Deck', y='Survived', data=all_data, palette='Set3')
plt.show()

#称呼与幸存的关系
all_data['Title'] = all_data['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
Title_Dict ={}
Title_Dict.update(dict.fromkeys(['Capt','Col','Major','Dr','Rev'],'Officer'))
Title_Dict.update(dict.fromkeys(['Don','Sir','the Countess','Dona','Lady'],'Royalty'))
Title_Dict.update(dict.fromkeys(['Mme','Ms','Mrs'],'Mrs'))
Title_Dict.update(dict.fromkeys(['Mlle','Miss'],'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'],'Mr'))
Title_Dict.update(dict.fromkeys(['Master','Jonkheer'],'Master'))
all_data['Title'] = all_data['Title'].map(Title_Dict)


plt.figure()
sns.barplot(x='Title',y='Survived',data=all_data, palette='Set3')
all_data['FamilySize'] = all_data['SibSp']+all_data['Parch']+1
plt.figure()
sns.barplot(x="FamilySize",y="Survived",data=all_data)

def Fam_label(s):
    if (s >= 2)& (s<= 4):
        return 2
    elif( (s>4)&(s<=7)) | (s==1):
        return 1
    elif(s>7):
        return 0


#数据的清晰与缺失值填充
#结合特征构造随机森林模型来填充Age

from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor
train = all_data[all_data['Survived'].notnull()]
test = all_data[all_data['Survived'].isnull()]

#分割数据，1:1
train_split_1, train_split_2 = cross_validation.train_test_split(train,test_size=0.5, random_state=0) #0.8  0.7

def predict_age_use_cross_validation(df1,df2,dfTest):
    age_df1 = df1[['Age','Pclass',"Sex","Title"]]
    age_df1 = pd.get_dummies(age_df1)
    age_df2 = df2[['Age','Pclass','Sex','Title']]
    age_df2 = pd.get_dummies(age_df2)

    known_age = age_df1[age_df1.Age.notnull()].as_matrix()

    unknow_age_df1 = age_df1[age_df1.Age.isnull()].as_matrix()
    unknow_age = age_df2[age_df2.Age.isnull()].as_matrix()
    print(unknow_age.shape)


    y = known_age[:,0]
    X = known_age[:,1:]

    rfr = RandomForestRegressor(random_state=0, n_estimators=1100, n_jobs=-1)
    rfr.fit(X,y)
    predictedAges = rfr.predict(unknow_age[:,1::])
    df2.loc[(df2.Age.isnull()),'Age'] = predictedAges
    predictedAges2 = rfr.predict(unknow_age_df1[:,1::])
    df1.loc[(df1.Age.isnull()),'Age'] = predictedAges2

    age_Test = dfTest[['Age','Pclass','Sex','Title']]
    age_Test = pd.get_dummies(age_Test)
    age_Tmp = df2[['Age','Pclass','Sex',"Title"]]
    age_Tmp = pd.get_dummies(age_Tmp)

    age_Tmp = pd.concat([age_Test[age_Test.Age.notnull()],age_Tmp])

    known_age1 = age_Tmp.as_matrix()
    unknow_age1 = age_Test[age_Test.Age.isnull()].as_matrix()
    y=known_age1[:,0]
    x = known_age1[:,1:]

    rfr.fit(x,y)
    predictedAges = rfr.predict(unknow_age1[:,1:])
    dfTest.loc[(dfTest.Age.isnull(),'Age')] = predictedAges

    return dfTest


t1 = train_split_1.copy()
t2 = train_split_2.copy()
tmp1 = test.copy()
t5 = predict_age_use_cross_validation(t1,t2,tmp1)
t1= pd.concat([t1,t2])

t3 = train_split_1.copy()
t4 = train_split_2.copy()
tmp2 = test.copy()
t6 = predict_age_use_cross_validation(t4, t3, tmp2)
t3 = pd.concat([t3, t4])

train['Age'] = (t1['Age'] + t3['Age']) / 2

test['Age'] = (t5['Age'] + t6['Age']) / 2
all_data = pd.concat([train, test])


all_data[all_data['Embarked'].isnull()]
all_data['Embarked'] = all_data['Embarked'].fillna('C')

#船票价格用中位数来填充

fare=all_data[(all_data['Embarked'] == "S") & (all_data['Pclass'] == 3)].Fare.median()
all_data['Fare']=all_data['Fare'].fillna(fare)


#同组识别
all_data['Surname']=all_data['Name'].apply(lambda x:x.split(',')[0].strip())
Surname_Count = dict(all_data['Surname'].value_counts())
all_data['FamilyGroup'] = all_data['Surname'].apply(lambda x:Surname_Count[x])
Female_Child_Group=all_data.loc[(all_data['FamilyGroup']>=2) & ((all_data['Age']<=12) | (all_data['Sex']=='female'))]
Male_Adult_Group=all_data.loc[(all_data['FamilyGroup']>=2) & (all_data['Age']>12) & (all_data['Sex']=='male')]
Female_Child=pd.DataFrame(Female_Child_Group.groupby('Surname')['Survived'].mean().value_counts())
Female_Child.columns=['GroupCount']
Female_Child
Male_Adult=pd.DataFrame(Male_Adult_Group.groupby('Surname')['Survived'].mean().value_counts())
Male_Adult.columns=['GroupCount']
Male_Adult
Female_Child_Group=Female_Child_Group.groupby('Surname')['Survived'].mean()
Dead_List=set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)
print(Dead_List)
Male_Adult_List=Male_Adult_Group.groupby('Surname')['Survived'].mean()
Survived_List=set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)
print(Survived_List)
train=all_data.loc[all_data['Survived'].notnull()]
test=all_data.loc[all_data['Survived'].isnull()]
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Sex'] = 'male'
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Age'] = 60
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Title'] = 'Mr'
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Sex'] = 'female'
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Age'] = 5
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Title'] = 'Miss'

all_data = pd.concat([train, test])
all_data = all_data[['Survived','Pclass','Sex','Age','Fare','Embarked','Title','FamilyLabel','Deck','TicketGroup']]
all_data = pd.get_dummies(all_data)

train = all_data[all_data['Survived'].notnull()]
test = all_data[all_data['Survived'].isnull().drop('Survived',axis=1)]
X = train.as_matrix()[:,1:]
y = train.as_matrix()[:,0]

from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV


#调参优化
pipe = Pipeline([('select',SelectKBest(k=20)),('classify',RandomForestClassifier(random_state= 10, max_features='sqrt'))])

param_test = {'classify_n_estimators':list(range(20,50,2)), 'classify_max_depth':list(range(3,60,3))}
gsearch = GridSearchCV(estimator=pipe, param_grid= param_test,scoring='roc_auc',cv=10)
gsearch.fit(X,y)


select = SelectKBest(k= 20)
clf = RandomForestClassifier(random_state=10, warm_start=True,
                             n_estimators=24,
                             max_depth=6,
                             max_features='sqrt')
pipeline =make_pipeline(select,clf)
pipeline.fit(X,y)
predictions = pipeline.predict(test)
submission = pd.DataFrame({'PassengerId':PassengerId,'Survived':predictions.astype(np.int32)})
submission.to_csv('trysubmission.csv')