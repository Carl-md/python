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

data=pd.read_csv("案例11数据.gzip",compression="gzip")
"""
# unique是列出每个值
# nunique是一共多少个值
print(data.country.nunique())
print(data.country.unique())
"""




