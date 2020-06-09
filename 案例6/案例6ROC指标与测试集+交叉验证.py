#      ┌─┐      ┌─┐
#   ┌──┘ ┴──────┘ ┴──┐
#   │                │
#   │  ─┬┘      └┬─  │
#   │                │
#   │      ─┴─       │
#   └───┐        ┌───┘
#       │        │
#       │        └──────────────┐
#       │                       ├─┐
#       │                       ┌─┘
#       └┐  ┐  ┌───────┬──┐  ┌──┘
#        │ ─┤ ─┤       │ ─┤ ─┤
#        └ ─┴──┘       └──┴──┘
#        神兽保佑      代码无BUG!
# -*- coding: UTF-8 -*-
import pandas as pd
from sklearn.linear_model import LogisticRegression

admissions=pd.read_csv("案例6数据.csv")
admissions["actual_label"]=admissions["admit"]
admissions=admissions.drop("admit",axis=1)
print(admissions.head())

import numpy as np
np.random.seed(8)

# 洗牌(打乱数据)
shuffled_index=np.random.permutation(admissions.index)
shuffled_admissions=admissions.loc[shuffled_index]

train=shuffled_admissions.iloc[0:515]
test=shuffled_admissions.iloc[515:len(shuffled_admissions)] # DataFrame行索引可以[a:b],但列索引不行，列只能一列一列的表示
# print(shuffled_admissions.head())
model=LogisticRegression()
model.fit(train[["gpa"]],train["actual_label"])
labels=model.predict(test[["gpa"]])
test["predicted_label"]=labels


# ROC曲线
import matplotlib.pyplot as plt
from sklearn import metrics

probabilities=model.predict_proba(test[["gpa"]])

fpr,tpr,thresholds=metrics.roc_curve(test["actual_label"],probabilities[:,1])
plt.plot(fpr,tpr)
plt.show()

# 求曲线面积得到积分值，积分值评价模型的好坏
auc_score=metrics.roc_auc_score(test["actual_label"],probabilities[:,1])
print(auc_score)
# 计算出来的auc_score即为预测正例和负例的综合评价结果
# 结果越趋近于1，分类效果越好

# 交叉验证
from sklearn.model_selection import KFold,cross_val_score
kf=KFold(5,shuffle=True,random_state=8)
IR=LogisticRegression()
accuracies=cross_val_score(IR,admissions[["gpa"]],admissions["actual_label"],scoring="roc_auc",cv=kf)
average_accuracy=sum(accuracies)/len(accuracies)
print(accuracies)
print(average_accuracy)