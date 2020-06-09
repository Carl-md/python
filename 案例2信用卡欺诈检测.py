import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
# %matplotlib inline
data = pd.read_csv("案例2数据.csv")
# print(data.head())
count_classses = pd.value_counts(data["Class"],sort=True).sort_index()
#运用value_counts计算class为0和1的个数
count_classses.plot(kind="bar")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()
# 对于样本不均衡数据
# 一种方法是下采样：让0和1样本数量一样小，使两个样本同样少
# 一种方法是过采样：让1和0样本数量一样多，使两个样本同样多
# 后续讨论两种采样方法哪个方法更适合

# 标准化处理：保证特征之间的分布差异是差不多的（同一量级）
# 导入标准化处理项
from sklearn.preprocessing import StandardScaler
# 标准化处理
data["normAmount"]=StandardScaler().fit_transform(data["Amount"].values.reshape(-1,1))
# fit_transform就是先拟合数据然后标准化
# reshape(-1,1)其中-1表示python自动判断行数

# 去除DataFrame中没有用的列
data=data.drop(["Time","Amount"],axis=1)

x=data.loc[:,data.columns != 'Class']
y=data.loc[:,data.columns == 'Class']
# 注意loc、iloc和ix的用法
# loc是行标签(标签可以取到最后一个值)
# iloc是行索引(索引取不到最后一个值)
# ix既可以是行序号，也可以是行标签(ix被弃用了)
number_records_fraud=len(data[data["Class"]==1])  # 欺诈的样本个数
fraud_indices=np.array(data[data["Class"]==1].index) # 欺诈样本的索引

normal_indices=data[data["Class"]==0].index

random_normal_indices=np.random.choice(normal_indices,number_records_fraud,replace=False)
# 从normal_indices中随机选取number_records_fraud个数
random_normal_indices=np.array(random_normal_indices)

under_sample_indices=np.concatenate([fraud_indices,random_normal_indices])
# concatenate将两者合并到一起

under_sample_data=data.iloc[under_sample_indices,:]

x_undersample = under_sample_data.loc[:,under_sample_data.columns!="Class"]
y_undersample = under_sample_data.loc[:,under_sample_data.columns=="Class"]

# x_undersample和y_undersample是下采样形成的样本

# 交叉验证
from sklearn.model_selection import train_test_split
# 整个数据集
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
# 下采样过后的数据集
x_train_undersample,x_test_undersample,y_train_undersample,y_test_undersample=train_test_split(x_undersample,y_undersample,test_size=0.3,random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import confusion_matrix,recall_score,classification_report
def print_kfold_scores(x_train_data, y_train_data):
    fold = KFold(5, shuffle=False)  # 数据数目，交叉折叠次数5次，不进行洗牌
    c_param_range = [0.01, 0.1, 1, 2, 3]  # 待选的模型参数
    # 新建一个dataframe类型，列名是参数取值、平均召回率
    # dataframe是一个五行二列的dataframe
    result_table = pd.DataFrame(index=range(len(c_param_range), 2),columns=['C_parameter', 'Mean Recall score'])
    result_table['C_parameter'] = c_param_range
    j = 0
    for c_param in c_param_range:
        print("----------------------------")
        print("C_parameter:", c_param)
        print("----------------------------")
        print("")

        recall_accs = []
        for iteration,indices in enumerate(fold.split(x_train_data)):
            lr = LogisticRegression(C=c_param, penalty='l2')  # 实例化逻辑回归模型
            lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())
            y_pred_undersample = lr.predict(x_train_data.iloc[indices[1], :].values)
            recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_pred_undersample)
            recall_accs.append(recall_acc)
            print("recall score=", recall_acc)
             # the mean value of the recall scores is the metric we want to save and fet hold of.
            result_table.loc[j, 'Mean Recall score'] = np.mean(recall_accs)
            j += 1
            print("")
            print("Mean Recall score:", np.mean(recall_accs))
        best_c = result_table.loc[result_table['Mean Recall score'].astype('float64').idxmax()]['C_parameter']
        # finally,we can check which C parameter is the best amongst the chosen
        print("**************************")
        print("Best model to choose from cross validation is with parameter= ", best_c)
        print("**************************")
    return best_c
best_c=print_kfold_scores(x_train_undersample,y_train_undersample)
# 混淆矩阵
# *******以上是下采样的方式
# 下面采用过采样的方式
"""
# 导入相应的库
from imblearn.over_sampling import SMOTE
# 过采样生成
oversampler=SMOTE(random_state=0) 
# random_state=0表示每次生成的都是相同的
os_x_train,os_y_train=oversampler.fit_sample(x_train,y_train) 
# 将实际数据x_train和y_train传进去,生成之后的数据集是os_x_train和os_y_train
"""
"""
总结：
一般建立模型数据越多越好
所以大部分不均衡模型都是采用过采样的方式
"""







