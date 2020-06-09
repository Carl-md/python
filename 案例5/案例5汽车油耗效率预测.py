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
columns=["mpg","cylinders","displacement","horsepower","weight","acceleration","model year","origin","car name"]
cars=pd.read_table("案例5数据.data",delim_whitespace=True,names=columns)
print(cars.head())
# 绘图
plt.figure(figsize=(10,8),dpi=80)
plt.subplot(2,1,1)
plt.scatter(cars["weight"],cars["mpg"])
plt.xlabel("weight")
plt.ylabel("mpg")
plt.subplot(2,1,2)
plt.scatter(cars["acceleration"],cars["mpg"])
plt.xlabel("acceleration")
plt.ylabel("mpg")
plt.show()
# 建立模型
from sklearn.linear_model import LinearRegression
Ir=LinearRegression(fit_intercept=True)
"""
# 按照列号取值
print(cars.iloc[:,[1,2,3,8]])
# 按照列名取值
print(cars[["cylinders","displacement","horsepower","weight","acceleration","model year","origin"]])
"""
Ir.fit(cars[["weight"]],cars["mpg"])
predictions=Ir.predict(cars[["weight"]])

# 确定衡量指标
# 计算MSE
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(cars["mpg"],predictions)
print(mse)
# 计算RMSE
rmse=(mse**(0.5))
print(rmse)