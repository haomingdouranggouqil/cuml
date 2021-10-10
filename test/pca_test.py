from sklearn import datasets
import time
from sklearn.decomposition import PCA as SPCA
import cupy as cp
from mlcu.ml.PCA import *


if __name__=="__main__":
    '''
    loaded_data = datasets.load_iris()    #加载鸢尾花数据
    data = cp.array(loaded_data.data)
    '''

    #此数据测试下cuml运算1.9s,sklearn13.1s
    X = cp.random.rand(1000000,100)
    pca = PCA(5)
    start_time = time.time()
    pca.fit(X)
    end_time = time.time()
    consum_time = end_time-start_time
    print(consum_time)


    X = X.tolist()
    start_time = time.time()
    spca = SPCA(n_components = 5)
    spca.fit(X)
    end_time = time.time()
    consum_time = end_time-start_time
    print(consum_time)
