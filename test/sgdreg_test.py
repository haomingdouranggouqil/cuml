from sklearn.linear_model import SGDRegressor
import time
import cupy as cp
from mlcu.ml.SGDREG import *



if __name__ == "__main__":
    # 生成数据x
    x = 5*cp.random.random((1000000,4))
    # 加入随机噪声
    x += cp.random.random(x.shape)
    label = cp.dot(x, cp.array([[-6.433333], [7.522222], [-6.433333], [7.522222]]) )
    # label 也加入随机噪声
    label += cp.random.random(label.shape)


    start_time = time.time()
    sgdreg = SGDREG()
    sgdreg.fit(x, label)
    end_time = time.time()
    consum_time = end_time-start_time
    print(consum_time)



    x = x.tolist()
    label = label.tolist()
    #sklearn
    start_time = time.time()
    l = SGDRegressor()
    l.fit(x, label)
    end_time = time.time()
    consum_time = end_time - start_time
    print(consum_time)
