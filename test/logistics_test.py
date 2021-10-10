from sklearn import datasets    #datasets模块
import cupy as cp
import numpy as np
import time
from mlcu.ml.LOGREG import *
from sklearn import linear_model



if __name__ == '__main__':
    '''
    loaded_data = datasets.load_iris()    #加载鸢尾花数据
    X = cp.array(loaded_data.data[:100]) #x有4个属性
    y = cp.array(loaded_data.target[:100])  #y 有2类
    '''

    #此测试下cuml运算4.9s，sklearn运算14s
    X = cp.random.rand(10000000,8)
    y = cp.random.randint(0,2,size=(10000000))

    lg = LogisticRegression()
    start_time = time.time()
    lg.fit(X, y, max_iter = 50)
    end_time = time.time()
    consum_time = end_time-start_time
    print(consum_time)


    X = X.tolist()
    y = y.tolist()
    start_time = time.time()
    logreg = linear_model.LogisticRegression(max_iter = 50)
    logreg.fit(X, y)
    end_time = time.time()
    consum_time = end_time-start_time
    print(consum_time)
