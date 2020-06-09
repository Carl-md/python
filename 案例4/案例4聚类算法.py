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

# 以下是KMeans聚类算法
import pandas as pd
from sklearn.preprocessing import StandardScaler

data=pd.read_csv("案例4数据.txt",sep=" ") # sep=" "表示以空格作为分隔符
x=data[["calories","sodium","alcohol","cost"]]

# 导入kmeans模块
from sklearn.cluster import KMeans
km=KMeans(n_clusters=3).fit(x)
km2=KMeans(n_clusters=2).fit(x)
x["cluster"]=km.labels_
x["cluster2"]=km2.labels_
x=x.sort_values(by="cluster")
x_grouped=x.groupby(by="cluster").mean()
print(x_grouped)

# 聚类之前需要先标准化
s=StandardScaler()
X=s.fit_transform(x)
km3=KMeans(n_clusters=3).fit(X)
x["scaler"]=km3.labels_
x["scaled_cluster"]=x["scaler"]
x=x.drop(["scaler"],axis=1)
print(x)

# 评价聚类结果的好坏——轮廓系数
# 导入模块
from sklearn import metrics
score_scale=metrics.silhouette_score(X,x.scaled_cluster)
score=metrics.silhouette_score(x,x.cluster)
print(score,score_scale)

# 以下是DBSCAN聚类算法
from sklearn.cluster import DBSCAN
db=DBSCAN(eps=10,min_samples=2).fit(x)
labels=db.labels_
x["cluster_db"]=labels
print(x)

