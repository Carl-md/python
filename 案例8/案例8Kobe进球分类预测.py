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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

raw=pd.read_csv("案例8数据.csv")

# 去除nan的行
kobe=raw[pd.notnull(raw["shot_made_flag"])]
# notnull函数去除了nan的行

import numpy as np
# 极坐标方式算距离和角度
raw["dist"]=np.sqrt(raw["loc_x"]**2+raw["loc_y"]**2)

loc_x_zero=raw["loc_x"]==0

raw["angle"]=np.array([0]*len(raw))
raw["angle"][~loc_x_zero]=np.arctan(raw["loc_y"][~loc_x_zero]/raw["loc_x"][~loc_x_zero])
raw["angle"][loc_x_zero]=np.pi*2
# 合并分钟和秒
raw["remaining_time"]=raw["minutes_remaining"]*60+raw["seconds_remaining"]
raw["season"]=raw["season"].apply(lambda x:int(x.split("-")[1]))
df=pd.DataFrame({"matchup":kobe.matchup,"opponent":kobe.opponent})

# plt.figure(figsize=(5,5))
# plt.scatter(raw.dist,raw.shot_distance,color="b")
# plt.show()

gs=kobe.groupby(by="shot_zone_area")

drops=["shot_id","team_id","team_name","shot_zone_area","shot_zone_range","shot_zone_basic","matchup",
       "lon","lat","seconds_remaining","minutes_remaining","shot_distance","loc_x","loc_y","game_event_id",
       "game_id","game_date"]

for drop in drops:
    raw=raw.drop(drop,axis=1)

categorical_vars=["action_type","combined_shot_type","shot_type","opponent","period","season"]
for var in categorical_vars:
    raw=pd.concat([raw,pd.get_dummies(raw[var],prefix=var)],1)
    # pd.get_dummies将其分为一列一列的DataFrame形式
    # pd.concat将分出来的一列一列的DataFrame按列合并到raw中
    raw=raw.drop(var,axis=1)

train_kobe=raw[pd.notnull(raw["shot_made_flag"])]
train_label=train_kobe["shot_made_flag"]
train_kobe=train_kobe.drop("shot_made_flag",axis=1)
test_kobe=raw[pd.isnull(raw["shot_made_flag"])]
test_kobe=test_kobe.drop("shot_made_flag",axis=1)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix,log_loss
import time

min_score=100000
best_n=0
scores_n=[]
range_n=np.logspace(0,2,num=3).astype(int)
for n in range_n:
    print("the number of trees:{0}".format(n))
    t1=time.time()

    rfc_score=0
    rfc=RandomForestClassifier(n_estimators=n)
    kf=KFold(10,shuffle=True)
    for train_k,test_k in kf.split(train_kobe):
        rfc.fit(train_kobe.iloc[train_k],train_label.iloc[train_k])
        pred=rfc.predict(train_kobe.iloc[test_k])
        rfc_score+=log_loss(train_label.iloc[test_k],pred)/10
    scores_n.append(rfc_score)
    if rfc_score<min_score:
        min_score=rfc_score
        best_n=n
    t2=time.time()
print(best_n,min_score)





