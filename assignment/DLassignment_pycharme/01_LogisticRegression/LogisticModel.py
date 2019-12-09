import numpy as np

'''
构建LogisticModel
'''
class LogisticModel():
    def __init__(self):
        pass

    #定义激活函数
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    #主要是初始化w，b
    def initiParams(self,dim):
        w = np.zeros((dim,1))
        b = 0.0

        assert(w.shape==(dim,1))
        assert(isinstance(b,float) or isinstance(b,int))

        return w,b
    #传播函数 得到梯度和cost
    def propogate(self,w,b,trainSet,trainLab):
        #正向传播
        Z = np.dot(w.T,trainSet)+b
        A = self.sigmoid(Z)
        #计算损失
        m = trainSet.shape[1]
        cost = -1/m *np.sum(trainLab * np.log(A) + (1-trainLab) * np.log(1-A))
        #根据损失计算梯度
        dw = 1/m *np.dot(trainSet,((A-trainLab).T))
        db = 1/m * np.sum(A - trainLab)

        assert (dw.shape == w.shape)
        assert (db.dtype == float)
        cost = np.squeeze(cost)
        assert (cost.shape == ())

        grads = {"dw": dw,
                 "db": db}
        #正向传播+反向传播后，返回梯度和cost
        return grads, cost

    #针对grads和cost做优化
    def optimize(self,w,b,trainSet,trainLab,num_iterations, learning_rate, print_cost = False):
        costs = [] #收集每一次训练的cost以便画图发现是否学到了东西
        #在叠在循环中
        for i in range(num_iterations):
            grads,cost = self.propogate(w,b,trainSet,trainLab) #先计算传播得到梯度和cost

            dw = grads['dw']
            db = grads['db']
            # 更新参数
            w = w - learning_rate * dw
            b = b - learning_rate * db

            # 记录数据
            if i % 100 == 0:
                costs.append(cost)

            # 每训练100次打印一次cost
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" % (i, cost))

            #记录权重参数和参数的梯度用于分析
        params = {
                'w':w,
                'b':b
            }

        grads = {
                'dw':dw,
                'db':db
            }
        #返回参数、梯度、loss值
        return params,grads,costs

    #定义预测函数
    def predict(self,w,b,X):
        m = X.shape[1]   # X一定要处理成列数=样本数
        Y_predict = np.zeros((1,m))
        Z = np.dot(w.T,X+b)
        A = self.sigmoid(Z)

        for i in range(m):
            if(A[0,i]>0.5):
                Y_predict[0,i] = 1
            else:
                Y_predict[0,i] = 0
        return Y_predict

    #综合起来成为一个大模型，输入训练集 标签 测试集 标签 迭代次数 学习率 就能得到一个结果

    def model(self,trainSet,trainLabel,testSet,testLabel,numIterations,learningRate):
        w,b = self.initiParams(trainSet.shape[0])
        params,grads,costs = self.optimize(w,b,trainSet,trainLabel,numIterations,learningRate,print_cost=True)

        w = params['w']
        b = params['b']

        Y_prediction_train = self.predict(w,b,trainSet)
        Y_prediction_test = self.predict(w,b,testSet)

        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - trainLabel)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - testLabel)) * 100))

        d = {"costs": costs,
             "Y_prediction_test": Y_prediction_test,
             "Y_prediction_train": Y_prediction_train,
             "w": w,
             "b": b,
             "learning_rate": learningRate,
             "num_iterations": numIterations}

        return d
