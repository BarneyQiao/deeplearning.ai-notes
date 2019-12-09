import numpy as np

'''
构建LogisticModel
'''
class LogisticModel():
    def __init__(self,trainSet,trainLab):
        self.trainSet = trainSet
        self.trainLab = trainLab
        self.dim = trainSet.shape[0] #维度

    #定义激活函数
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    #主要是初始化w，b
    def initiParams(self):
        self.w = np.zeros((self.dim,1))
        self.b = 0.0

        assert(self.w.shape==(self.dim,1))
        assert(isinstance(self.b,float) or isinstance(self.b,int))

        return self.w,self.b

    def propogate(self):
        #正向传播
        Z = np.dot(self.w.T,self.trainSet)+self.b
        A = self.sigmoid(Z)
        #计算损失
        m = self.trainSet.shape[1]
        cost = -1/m *np.sum(self.trainLab * np.log(A) + (1-self.trainLab) * np.log(1-A))
