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
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pylab import style
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

stockFile="案例9数据.csv"
stock=pd.read_csv(stockFile,index_col=0,parse_dates=[0]) # index_col将第1列作为索引列
# print(stock.head())
stock_week=stock["Close"].resample("W-MON").mean()
stock_train=stock_week["2000":"2015"]

# 做一阶差分
stock_diff=stock_train.diff()
stock_diff=stock_diff.dropna()

# 画出acf和pacf的图
acf=plot_acf(stock_diff,lags=20)
plt.title("ACF")
plt.show()

pacf=plot_pacf(stock_diff,lags=20)
plt.title("PACF")
plt.show()

# 训练模型
model=ARIMA(stock_train,order=(1,1,1),freq="W-MON")
result=model.fit()
pred=result.predict("20140609","20160701",dynamic=True,typ="levels")
print(pred)
