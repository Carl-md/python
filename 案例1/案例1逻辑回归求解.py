import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
path = "案例1数据.txt"
pddata=pd.read_csv(path,names=["Exam 1","Exam 2","admitted"])
# print(pddata.head())
# print(pddata.shape)
#绘制0和1的散点图
positive=pddata[pddata["admitted"]==1]
negative=pddata[pddata["admitted"]==0]
plt.figure(figsize=(10,8),dpi=80)
plt.scatter(positive["Exam 1"],positive["Exam 2"],s=30,c="b",marker="o",label="admitted")
plt.scatter(negative["Exam 1"],negative["Exam 2"],s=30,c="r",marker="x",label="not admitted")
plt.legend(loc="best",fontsize=15)
plt.xlabel("Exam 1 score",fontsize=15)
plt.ylabel("Exam 2 score",fontsize=15)
plt.show()
#建立分类器
#定义sigmoid函数
def sigmoid(z):
    return 1/(1+np.exp(-z))
#定义输入变量
def model(x,theta):
    return sigmoid(np.dot(x,theta.T))#.T是简单转置的意思
#插入一列1的数据
pddata.insert(0,"ones",1)
#设置x和y
origin_data=pddata.values
cols=origin_data.shape[1]
x=origin_data[:,0:cols-1]
y=origin_data[:,cols-1:cols]
theta=np.zeros([1,3])
#定义损失函数
def cost(x,y,theta):
    left=np.multiply(-y,np.log(model(x,theta)))
    right=np.multiply(1-y,np.log(1-model(x,theta)))
    return np.sum(left-right)/(len(x))
#定义梯度
def gradient(x,y,theta):
    grad=np.zeros([theta.shape[0],theta.shape[1]])
    error=(model(x,theta)-y).ravel() # ravel()展开成1行多列
    for j in range(len(theta.ravel())):
        term=np.multiply(error,x[:,j])
        grad[0,j]=np.sum(term)/len(x)
    return grad
stop_iter=0
stop_cost=1
stop_grad=2
def stopCriterion(type,value,threshold):
    #设定三种不同的停止策略
    if type == stop_iter: return value>threshold
    elif type == stop_cost: return abs(value[-1]-value[-2])<threshold
    elif type == stop_grad: return np.linalg.norm(value)<threshold# 返回大的值，比较前后两个值，谁大谁输出
import numpy.random
#定义洗牌
def shuffleData(data):
    np.random.shuffle(data) # 自动打乱
    cols=data.shape[1]
    x=data[:,0:cols-1]
    y=data[:,cols-1:]
    return x,y
import time
def descent(data,theta,batchsize,stopType,thresh,alpha):
    #梯度下降求解
    init_time=time.time()
    i=0#迭代次数
    k=0#batch
    x,y=shuffleData(data)
    grad=np.zeros(theta.shape)
    costs=[cost(x,y,theta)]#计算损失值

    while True:
        grad=gradient(x[k:k+batchsize],y[k:k+batchsize],theta)
        k+=batchsize
        if k>=n:
            k=0
            x,y=shuffleData(data)#参数洗牌
        theta=theta-alpha*grad#更新参数
        costs.append(cost(x,y,theta))#计算损失值
        i+=1
        if stopType==stop_iter:   value=i
        elif stopType==stop_cost: value=costs
        elif stopType==stop_grad: value=grad
        if stopCriterion(stopType,value,thresh):break
    return theta,i-1,costs,grad,time.time() - init_time
n=50
descent(origin_data,theta,n,stop_cost,thresh=0.001,alpha=0.001)
print(descent(origin_data,theta,n,stop_cost,thresh=0.001,alpha=0.001))
