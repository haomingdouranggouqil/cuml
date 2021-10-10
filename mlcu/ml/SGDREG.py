import cupy as cp

class SGDREG:
    #梯度下降回归模型
    def __init__(self):
        self.train_data = None
        self.train_label = None
        self.alpha = 0.01
        self.num_iters = 500
        self.coef = None
        self.test = None

    def fit(self, train_data, train_label, alpha = 0.01, num_iters = 500):
        self.train_data = cp.array(train_data)
        self.train_label = cp.array(train_label)
        self.alpha = alpha
        self.num_iters = num_iters
        m = len(self.train_label)                    # 总的数据条数
        col = len(self.train_data[0])               # 特征数
        theta = cp.zeros((col, 1))
        self.train_label = self.train_label.reshape(-1, 1)           #将行向量转化为列
        theta = self.gradientDescent(theta)
        self.coef = theta
        return theta                  #返回学习的结果theta

    def gradientDescent(self, theta):
        m = len(self.train_label)
        n = len(theta)
        temp = cp.zeros((n, self.num_iters))  # 暂存每次迭代计算的theta，转化为矩阵形式
        for i in range(self.num_iters):  # 遍历迭代次数
            h = cp.dot(self.train_data, theta)     # 计算内积，matrix可以直接乘
            e = cp.dot(cp.transpose(self.train_data), h - self.train_label)
            t = theta - ((self.alpha / m) * e)   #梯度的计算
            temp[:, i] = t.flatten()
            theta = t
        return theta

    def predict(self, t):
        self.test = cp.array(t)
        return cp.dot(self.test, self.coef)
