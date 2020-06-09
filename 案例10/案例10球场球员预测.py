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
#        神兽保佑     代码永无BUG!
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("案例10数据.csv")
# print(df.shape)
# print(df.describe().T)
# describe可以帮助我们粗略的看一下分布
# 包括数量、均值、标准差、最大值、最小值和四分位数
# T为转置
# print(df.dtypes)
# object为string类型，python识别不出来
all_columns=df.columns.tolist()
# print(all_columns)
# d=np.mean(df.groupby(by="playerShort").height.mean())
# print(d)
player_index="playerShort"
player_cols=["birthday","height","weight","position","photoID","rater1","rater2"]
all_cols_unique_players=df.groupby(by="playerShort").agg({col:"nunique" for col in player_cols})
# 统计每个playerShort出现的次数（正常只会出现一次）
# print(all_cols_unique_players)
print(all_cols_unique_players[all_cols_unique_players>1].dropna().head())
# 证明数据中没有重复的playerShort

players=df[player_cols]
"""
import missingno as msno
msno.matrix(players.sample(10000))
msno.heatmap(players.sample(500))
缺失值统计
matrix有空白即为有缺失值
plt.show()
"""
# 也可以运用isnull和notnull函数来查询是否具有缺失值
# print(pd.crosstab(players.rater1,players.rater2))
"""
# 输出所有的位置
position_types=players.position.unique()
print(position_types)
"""
"""
# 将体重均分为几个级别
weight_categories=["vlow_weight","low_weight","mid_weight","high_weight","vhigh_weight"]
players["weightclass"]=pd.qcut(players["weight"],len(weight_categories),weight_categories)
print(players.head())
"""
# print(players.birthday.head())
# 将日期调成规范的格式
# 并计算每个球员的年龄
players["birth_date"]=pd.to_datetime(players.birthday)
players["age_years"]=((pd.to_datetime("2020-4-4")-players["birth_date"]).dt.days)//365
print(players.head())
