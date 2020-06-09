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
import matplotlib.pyplot as plt
#导入数据
columns=["mpg","cylinders","displacement","horsepower","weight","acceleration","year","origin","car name"]
cars=pd.read_table("案例7数据.data",delim_whitespace=True,names=columns)
# print(cars.head())
dumy_cylinders=pd.get_dummies(cars["cylinders"],prefix="cy1")
cars=pd.concat([cars,dumy_cylinders],axis=1)
dumy_years=pd.get_dummies(cars["year"],prefix="year")
cars=cars.drop("year",axis=1)
cars=cars.drop("cylinders",axis=1)
# print(cars.head())

# 划分训练集和测试集
import numpy as np
shuffled_rows=np.random.permutation(cars.index)
shuffled_cars=cars.iloc[shuffled_rows]
highest_train_row=int(cars.shape[0]*0.70)
# 取70%的数据当训练集，剩下30%的数据当测试集
train=shuffled_cars.iloc[0:highest_train_row]
test=shuffled_cars.iloc[highest_train_row:]

from sklearn.linear_model import LogisticRegression

unique_origins=cars["origin"].unique()
unique_origins.sort()

models=[]
features=[c for c in train.columns if c.startswith("cy1") or c.startswith("year")]

for origin in unique_origins:
    model=LogisticRegression()

    X_train = train[features]
    Y_train = train["origin"]==origin

    model.fit(X_train,Y_train)

    models.append(model)

testing_probs=pd.DataFrame(columns=unique_origins)

for origin in unique_origins:
    X_test=test[features]
    testing_probs[origin].append(models[origin].predict_proba(X_test)[:,1])

predicted_origins=testing_probs.idxmax(axis=1)
print(predicted_origins)
# one vs all
