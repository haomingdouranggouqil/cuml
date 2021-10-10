# cuml
## An incomplete machine learning library which supporting GPU
#### 目前尚不完善，将来拟增加AI/NLP/CV相关算法
## 在大量数据环境下，cuml的速度会是sklearn相同算法的数倍甚至数十倍。
#### 环境：Windows10，python3.9，NVIDIA 3060
|algorithm|scale |mlcu  |sklearn|
| ---- | ---- | ---- |----   |
|KMeans |(100000,4) | 72s  | 195s  |
|KNN |(100000,4) | 12s  | 31s  |
|LinearRegression |(10000000,4) | 0.9s  | 12.6s  |
|LogisticRegression |(10000000,8)| 4.9s | 14s |
|NaiveBayes |(10000000,10)|0.48s| 24.26s  |
|pca |(1000000,100) | 1.9s  | 13.1s |
|SGDRegressor |(1000000,4)| 0.9s  | 2.7s |
#### 当然，在数据量较小的情况下，由于将数据传输至显卡所需的额外时间消耗，mlcu的表现要较sklearn差。
#### 之前需要用机器学习算法处理大量数据，想用显卡加速，结果发现sklearn不能调用显卡，而GitHub和pypi上也找不到一个相关的包，于是没办法只能自己动手了，暂时完成了这几个常用的算法，大数据下能做到比sklearn快数倍或数十倍。由于这可能是世界上第一个覆盖主流机器学习算法的python框架，而且编写时间只有几天，所以非常不完善，目前只支持这几种算法，以后会考虑增加AI/NLP/CV相关算法练练手
