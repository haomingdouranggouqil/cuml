import cupy as cp
import time
from sklearn.cluster import KMeans
from mlcu.ml.KMEANS import *

if __name__ == "__main__":
    # 生成数据x
    #x = 5 * cp.random.random((10000000,4))
    #此数据量下，cuml运算72s,sklearn运算195s
    x = 5 * cp.random.random((100000,4))

    start_time = time.time()
    km = KMEANS(4)
    km.fit(x)
    print(km.centers)
    end_time = time.time()
    consum_time = end_time-start_time
    print(consum_time)

    x = x.tolist()
    start_time = time.time()
    kmeans = KMeans(n_clusters = 4)
    kmeans.fit(x)
    print(kmeans.cluster_centers_)
    end_time = time.time()
    consum_time = end_time-start_time
    print(consum_time)
