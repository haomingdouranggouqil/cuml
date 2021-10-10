import cupy as cp

class PCA():
   def __init__(self, n_dim):
       #降到k维
       self.n_dim = n_dim
       self.mean_vector = None
       self.feature_vector_mat = None
       self.pca_data = None
       self.restruct_data = None
   #数据标准化（去均值）
   def stand_data(self,data):
       #axis = 0,按列取值求均值
       self.mean_vector = cp.mean(data,axis=0)
       return self.mean_vector, data - self.mean_vector

   # 计算协方差矩阵
   def getCovMat(self,standData):
       # rowvar=0表示数据的每一列代表一个维度
       return cp.cov(standData,rowvar=0)

   # 计算协方差矩阵的特征值和特征向量
   def getFValueAndFVector(self,covMat):
       fValue,fVector = cp.linalg.eigh(covMat)
       return fValue,fVector

   # 得到特征向量矩阵
   def getVectorMatrix(self,fValue,fVector):
       #从大到小排序，并返回排序后的原索引值
       fValueSort = cp.argsort(-fValue)
       #print(fValueSort)
       fValueTopN = fValueSort[:self.n_dim]
       #print(fValueTopN)
       self.feature_vector_mat = fVector[:,fValueTopN]
       return self.feature_vector_mat

   # 得到降维后的数据
   def fit(self,data):
       (mean_vector ,standdata)= self.stand_data(data)
       cov_mat = self.getCovMat(standdata)
       fvalue,fvector = self.getFValueAndFVector(cov_mat)
       fvectormat = self.getVectorMatrix(fvalue,fvector)
       self.pca_data = cp.dot(standdata,fvectormat)
       return self.pca_data

   def restruct(self):
       self.restruct_data = cp.matmul(self.pca_data, self.feature_vector_mat.T) + self.mean_vector
       return self.restruct_data
