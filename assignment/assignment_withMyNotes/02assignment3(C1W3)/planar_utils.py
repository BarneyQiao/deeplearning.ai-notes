import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model

# 画决策边界，很炫的技能，也就是说将一个区域内的网格点以预测值为高度，然后用函数把相同高度的点区域内着色。多看几遍。
def plot_decision_boundary(model, X, y):    # 输入参数为三个，预测值 y· ，X，真实值y
    # X 的维度为: (2, 400)   Y 的维度为: (1, 400)
    # Set min and max values and give it some padding 取得这些点的一个范围
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1  # 取400个Example中横坐标最小的，最大的
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1  # 取400个Example中纵坐标最小的，最大的
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    '''
    一句话解释numpy.meshgrid()——生成网格点坐标矩阵。用这种方法画决策边界。
    返回值： np.arange()函数返回一个有终点和起点的固定步长的排列，如[1,2,3,4,5]，起点是1，终点是5，步长为1。 
    参数个数情况： np.arange()函数分为一个参数，两个参数，三个参数三种情况 
    1）一个参数时，参数值为终点，起点取默认值0，步长取默认值1。 
    2）两个参数时，第一个参数为起点，第二个参数为终点，步长取默认值1。 
    3）三个参数时，第一个参数为起点，第二个参数为终点，第三个参数为步长。其中步长支持小数。
    这里的用法是生成一个x范围为min-max，y范围为min-max 步长为0.1，再用meshgrid函数生成网格矩阵（输出值） xx yy
    '''
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral) 
    '''
    contourf函数是绘制等高线的函数，等高线的绘制主要是三个要素 x，y和（x,y）的高度
    这里把xx中的作为x坐标
    yy中的作为y坐标
    Z就是预测对应（x,y）点的logistic回归预测值（只有0或者1）
    https://www.jb51.net/article/130617.htm
    plt.contourf 与 plt.contour 区别：

    f：filled，也即对等高线间的填充区域进行填充（使用不同的颜色）；
    contourf：将不会再绘制等高线（显然不同的颜色分界就表示等高线本身），
    '''
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
   # x.astype(np.float128)
    s = 1/(1+np.exp(-x))
    return s

def load_planar_dataset():
    np.random.seed(1) #设定random.seed(1)
    '''
    如何理解random.seed(参数)，其实就是为了设定一个种子(参数),参数值不重要，但是相同参数下生成的随机数是一样的
    但是注意seed是有作用域的
    '''
    m = 400 # number of examples
    N = int(m/2) # number of points per class  这里是说每个example是由两个点组成，因此在每一个类别中有200个点
    D = 2 # dimensionality 维度2
    X = np.zeros((m,D)) # (400,2)data matrix where each row is a single example  样本集就是一个（样本数量，二维点）[[坐标1，坐标2]，....]
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue) 标签集是一个样本数量为行，结果为列的，也就是一个(400,1)的列向量
    a = 4 # maximum ray of the flower  也就是规定了组成点坐标的最大值不超过4

    #这里将如何生成
    for j in range(2):  # 设置range(2)是为了 正反例都是200个用2个循环去生成 
        ix = range(N*j,N*(j+1))       # 两次循环 第一次range(0,200)  第二次(200,400)
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
        '''
        np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)  
        在start 到stop的范围内等距生成num个点（默认为50），包括stop，这里生成了N=200个点
        
        numpy.random.randn(d0,d1,…,dn)
        randn函数返回一个或一组样本，具有标准正态分布。
        dn表格每个维度
        返回值为指定维度的array
        '''
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]  #np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。这里就是把俩个数组成一个行矩阵，赋值给X的第ix行
        Y[ix] = j  #生成该类的标签
        
    X = X.T   # 转置形成(2,400)样本集
    Y = Y.T   # 转置形成(1,400)的标签集

    return X, Y

def load_extra_datasets():  
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)
    
    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure