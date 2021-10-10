import cupy as cp

class KMEANS:
    #kmeans模型聚类
    def __init__(self, k):
        self.train_data = None
        self.k = k
        self.centers = None
        self.clusters = None
        self.test = None
        self.seed = None
        self.tolerance = None
        self.max_iter = None

    def distance(self, vector1, vector2):
        return cp.sqrt(cp.sum(cp.square(vector2 - vector1)))

    # 初始化k个中心点
    def init_centroids(self):
        cp.random.seed(self.seed)
        n = self.train_data.shape[0]
        assert n >= self.k
        idxs = cp.random.choice(range(n), self.k, replace=False)
        return self.train_data[idxs]

    # 计算k个cluster的中心点
    def compute_centroids(self):
        new_centroids = cp.zeros((self.k, self.train_data.shape[1]))
        for i in range(self.k):
            new_centroids[i] = cp.mean(self.train_data[self.clusters == i], axis=0)
        return new_centroids

    # 计算所有样本点和中心点的距离
    def compute_distances(self):
        double_xy = 2 * self.train_data.dot(self.centers.T)
        sq_X = cp.sum(cp.square(self.train_data), axis=1, keepdims=True)
        sq_centers = cp.sum(cp.square(self.centers), axis=1)
        dists = cp.sqrt(abs(sq_X - double_xy + sq_centers))

        return dists

    def fit(self, train_data, seed = 110, tolerance = 1e-5, max_iter = 1000):
        self.train_data = train_data
        self.seed = seed
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.centers = self.init_centroids()
        dists = self.compute_distances()
        self.clusters = cp.argmin(dists, axis=1)

        i = 0
        while True:
            # 重新计算中心点
            new_centers = self.compute_centroids()
            if i > max_iter or self.distance(new_centers, self.centers) <= tolerance:
                break
            self.centers = new_centers
            dists = self.compute_distances()
            self.clusters = cp.argmin(dists, axis=1)
            i += 1
        return self.clusters, self.centers
