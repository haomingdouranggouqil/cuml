from mlcu.ml.KNN import *
from sklearn import datasets    #datasets模块
from sklearn.model_selection import train_test_split    #分离训练集和测试集数据
from sklearn.neighbors import KNeighborsClassifier    #k近邻分类器模块
from sklearn.neighbors import KNeighborsRegressor
import time

if __name__ == '__main__':
    #测试数据
    train_data = [[1, 1, 1], [2, 2, 2], [10, 10, 10], [13, 13, 13]]
    #数字标签
    train_label = [1, 2, 30, 60]
    #非数字标签
    #train_label = ['aa', 'aa', 'bb', 'bb']
    #测试数据
    test_data = [[3, 2, 4], [9, 13, 11], [10, 20, 10]]
    #默认分类
    knn = KNN()
    #knn回归任务
    #knn = KNN(task_type = 'regression')
    #训练
    knn.fit(train_data, train_label)
    #预测
    preds = knn.predict(test_data, k=2)
    print(preds)



    '''
    #鸢尾花测试
    #准备数据
    loaded_data = datasets.load_iris()    #加载鸢尾花数据
    X = loaded_data.data #x有4个属性
    y = loaded_data.target #y 有三类
    '''

    '''
    #生成大数据测试
    X = np.random.rand(100000,4).tolist()
    y = np.random.randint(1,4,size=(100000)).tolist()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)#测试数据占30%
    #在此数据量下，cuml模型运算12s，用了相同算法（暴力搜索）的sklearn运算了31s，numpy模型运算192s
    '''

    '''
    #cuml
    knn = KNN(task_type = 'regression')
    knn.fit(X_train, y_train)
    start_time = time.time()
    preds = knn.predict(X_test)
    end_time = time.time()
    consum_time = end_time-start_time
    print(consum_time)
    '''

    '''
    knn = KNeighborsRegressor(algorithm = 'brute') #k近邻分离器
    knn.fit(X_train, y_train)    #fit学习函数
    start_time = time.time()
    knn.predict(X_test)
    end_time = time.time()
    consum_time = end_time-start_time
    print(consum_time)
    '''
