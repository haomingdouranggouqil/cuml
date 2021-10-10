import cupy as cp
import cuml.common.func as func

class LogisticRegression:
    def __init__(self, penalty="l2", gamma=0, fit_intercept=True):
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg
        self.beta = None
        self.gamma = gamma
        self.penalty = penalty
        self.fit_intercept = fit_intercept

    def fit(self, X, y, lr=0.01, tol = 1e-4, max_iter = 1e5):
        if self.fit_intercept:
            X = cp.c_[cp.ones(X.shape[0]), X]

        l_prev = cp.inf
        self.beta = cp.random.rand(X.shape[1])
        for _ in range(int(max_iter)):
            y_pred = func.sigmoid(cp.dot(X, self.beta))
            loss = self._NLL(X, y, y_pred)
            if l_prev - loss < tol:
                return
            l_prev = loss
            self.beta -= lr * self._NLL_grad(X, y, y_pred)

    def _NLL(self, X, y, y_pred):
        N, M = X.shape
        beta, gamma = self.beta, self.gamma
        order = 2 if self.penalty == "l2" else 1
        norm_beta = cp.linalg.norm(beta, ord=order)

        nll = -cp.log(y_pred[y == 1]).sum() - cp.log(1 - y_pred[y == 0]).sum()
        penalty = (gamma / 2) * norm_beta ** 2 if order == 2 else gamma * norm_beta
        return (penalty + nll) / N

    def _NLL_grad(self, X, y, y_pred):
        N, M = X.shape
        l1norm = lambda x: cp.linalg.norm(x, 1)  # noqa: E731
        p, beta, gamma = self.penalty, self.beta, self.gamma
        d_penalty = gamma * beta if p == "l2" else gamma * cp.sign(beta)
        loss = y - y_pred
        tc = cp.dot(loss, X)
        return -(tc + d_penalty) / N

    def predict(self, X):
        if self.fit_intercept:
            X = cp.c_[cp.ones(X.shape[0]), X]
        return func.sigmoid(cp.dot(X, self.beta))
