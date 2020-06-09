import pandas as pd

# 导入数据
titanic=pd.read_csv("案例3数据_train.csv")

# 数据预处理之nan处理
titanic["Age"]=titanic["Age"].fillna(titanic["Age"].median())
# 用中位数填充Age列中的nan值

# 原始数据中的性别是男，女
# 需要将其转化为数值的量，机器学习才能识别
# 转化过程如下：
titanic.loc[titanic["Sex"]=="male","Sex"]=0
titanic.loc[titanic["Sex"]=="female","Sex"]=1
# 上船地点转化为数值
titanic["Embarked"]=titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"]=="S","Embarked"]=0
titanic.loc[titanic["Embarked"]=="C","Embarked"]=1
titanic.loc[titanic["Embarked"]=="Q","Embarked"]=2

# 利用回归算法预测
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

predictors=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]

alg=LinearRegression()

kf=KFold(n_splits=5,random_state=None)
prediction=[]
for train,test in kf.split(titanic):
    train_predictors=(titanic[predictors].iloc[train,:])
    train_target=titanic["Survived"].iloc[train]
    alg.fit(train_predictors,train_target)
    test_predictors=(titanic[predictors].iloc[test,:])
    test_prediction=alg.predict(test_predictors)
    prediction.append(test_prediction)

# 交叉验证出来的是五组数据
# 将五组数据进行合并评估模型预测精度
import numpy as np

predictions=np.concatenate(prediction,axis=0)

predictions[predictions >= 0.5]=1
predictions[predictions < 0.5]=0

accuracy=sum(predictions[predictions==titanic["Survived"]])/len(predictions)
# sum是求和，如果用len的话True和False都会被算进去
print(accuracy)
"""
# 运用测试集来测试
titanic_test=pd.read_csv("案例3数据_test.csv")

# 数据预处理之nan处理
titanic_test["Age"]=titanic_test["Age"].fillna(titanic_test["Age"].median())
# 用中位数填充Age列中的nan值
titanic_test["Fare"]=titanic_test["Fare"].fillna(titanic_test["Fare"].median())
# 原始数据中的性别是男，女
# 需要将其转化为数值的量，机器学习才能识别
# 转化过程如下：
titanic_test.loc[titanic_test["Sex"]=="male","Sex"]=0
titanic_test.loc[titanic_test["Sex"]=="female","Sex"]=1
# 上船地点转化为数值
titanic_test["Embarked"]=titanic_test["Embarked"].fillna("S")
titanic_test.loc[titanic_test["Embarked"]=="S","Embarked"]=0
titanic_test.loc[titanic_test["Embarked"]=="C","Embarked"]=1
titanic_test.loc[titanic_test["Embarked"]=="Q","Embarked"]=2
"""
# 以下运用随机森林来建立分类器
print("运用随机森林建立分类器中......")

# 导入模块
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

# 建立随机森林分类器
alg_rf=RandomForestClassifier(random_state=1,n_estimators=50,min_samples_split=4,min_samples_leaf=2)
kf_1=model_selection.KFold(n_splits=5,random_state=None)
scores=model_selection.cross_val_score(alg_rf,titanic[predictors],titanic["Survived"],cv=kf_1)
print(scores.mean())
# 提升分类器精确度：增加特征
"""
# 随机森林特征重要性分析
# 向特征中添加噪音，比较前后的error，显示出特征的重要性
"""
import numpy as np
from sklearn.feature_selection import SelectKBest,f_classif
import matplotlib.pyplot as plt

selector=SelectKBest(f_classif,k=5)
selector.fit(titanic[predictors],titanic["Survived"])

scores_1=-np.log10(selector.pvalues_)

plt.bar(range(len(predictors)),scores_1)
plt.xticks(range(len(predictors)),predictors,rotation="vertical")
plt.show()
# 柱状图的柱越高该特征越重要

