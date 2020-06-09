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
from matplotlib import pyplot as plt

admissions=pd.read_csv("案例6数据.csv")
print(admissions.head())
plt.scatter(admissions["gpa"],admissions["admit"])
plt.show()

import numpy as np

def logit(x):
    return np.exp(x)/(1+np.exp(x))

# x=np.linspace(-6,6,50,dtype=float)
# y=logit(x)
# plt.plot(x,y)
# plt.ylim(0,1)
# plt.xlim(-6,6)
# # plt.grid()
# plt.show()
"""
from sklearn.linear_model import LinearRegression
linear_model=LinearRegression()
linear_model.fit(admissions[["gpa"]],admissions["admit"])
"""
from sklearn.linear_model import LogisticRegression
logistic_model=LogisticRegression()
logistic_model.fit(admissions[["gpa"]],admissions["admit"])
pred_probs=logistic_model.predict(admissions[["gpa"]])
print(pred_probs)
plt.scatter(admissions[["gpa"]],pred_probs)# 1表示第二列即被接收的概率
plt.show()

admissions["predicted_label"]=pred_probs
admissions["actual_label"]=admissions["admit"]

# 确定衡量标准
"""
# 运用accuracy来评价模型预测精度
matches = (admissions["predicted_label"] == admissions["actual_label"])
correct_predictions=admissions[matches]
accuracy=len(correct_predictions)/float(len(admissions)) # float是将整数和字符串转换成浮点数
print(accuracy)
"""
"""
TPR=True Positives/(True Positives + False Negatives)
TPR=(1预测为1的样本量)/((1预测为1的样本量)+(1预测为0的样本量))
"""
# 运用TPR来评价模型预测精度
true_positive_filter=(admissions["predicted_label"]==1) & (admissions["actual_label"]==1)
true_positives=len(admissions[true_positive_filter])

true_negative_filter=(admissions["predicted_label"]==0) & (admissions["actual_label"]==0)
true_negatives=len(admissions[true_negative_filter])

false_positive_filter=(admissions["predicted_label"]==1) & (admissions["actual_label"]==0)
false_positives=len(admissions[false_positive_filter])

false_negative_filter=(admissions["predicted_label"]==0) & (admissions["actual_label"]==1)
false_negatives=len(admissions[false_negative_filter])

TRP=true_positives/float(true_positives+false_negatives) # TRP计算的为检测正例的效果
TNP=true_negatives/float(true_negatives+false_positives)

print(TRP)
print(TNP)




