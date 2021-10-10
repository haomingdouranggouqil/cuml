import cupy as cp

class LINEAR:
    #LINEAR模型，用最小二乘法，支持岭回归
    def __init__(self):
        self.train_data = None
        self.train_label = None
        self.coef = None
        self.test = None
        self.l1_ratio = None
        self.l2_ratio = None

    def fit(self, train_data, train_label, l1_ratio = 0, l2_ratio = 0):
        self.train_data = train_data
        self.train_label = train_label
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        if self.l1_ratio == 0 and self.l2_ratio == 0:
            self.coef = cp.dot(cp.dot(cp.linalg.inv(cp.dot(self.train_data.T, self.train_data)), self.train_data.T), self.train_label)
        elif self.l1_ratio == 0 and self.l2_ratio != 0:
            self.coef = cp.linalg.inv(self.train_data.T.dot(self.train_data) + self.l2_ratio * cp.eye(self.train_data.shape[1])).dot(self.train_data.T).dot(self.train_label)
        elif self.l1_ratio != 0 and self.l2_ratio == 0:
            print("lasso暂不支持")
        else:
            print("elasticnet暂不支持")

    def predict(self, t):
        self.test = cp.array(t)
        return cp.dot(self.test, self.coef)
