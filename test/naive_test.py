import cupy as cp
from sklearn import datasets
from sklearn import naive_bayes
import time
from mlcu.ml.NB import *


if __name__ == '__main__':
    X = cp.random.rand(10000000,10)
    y = cp.random.randint(0, 4,size=(10000000))

    #此数据下测试，cuml运行0.48s，sklearn运行24.26s
    start_time = time.time()
    gnb = GaussianNBClassifier()
    gnb.fit(X, y)
    end_time = time.time()
    consum_time = end_time-start_time
    print(consum_time)


    X = X.tolist()
    y = y.tolist()
    start_time = time.time()
    clf = naive_bayes.GaussianNB()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    end_time = time.time()
    consum_time = end_time-start_time
    print(consum_time)
