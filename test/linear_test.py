from sklearn.linear_model import LinearRegression
import time
import cupy as cp
from mlcu.ml.LINEAR import *


if __name__ == "__main__":
    # 生成数据x
    x = 5*cp.random.random((10000000,4))
    # 加入随机噪声
    x += cp.random.random(x.shape)
    label = cp.dot(x, cp.array([[-6.433333], [7.522222], [-6.433333], [7.522222]]) )
    # label 也加入随机噪声
    label += cp.random.random(label.shape)
    #在此数据量下，cuml运算0.9s, sklearn运算12.6s


    start_time = time.time()
    lin = LINEAR()
    lin.fit(x, label, l2_ratio = 1)
    print(lin.predict(x[:5]))
    print(label[:5])
    end_time = time.time()
    consum_time = end_time-start_time
    print(consum_time)


    x = x.tolist()
    label = label.tolist()

    start_time = time.time()
    l = LinearRegression()
    l.fit(x, label)
    end_time = time.time()
    consum_time = end_time - start_time
    print(consum_time)
