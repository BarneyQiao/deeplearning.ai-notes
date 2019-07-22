import numpy as np

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))  #根据本层计算出来的Z 来计算sigmoid值
    cache = Z     # 把本层的Z计入下来，用于反向传播计算
    
    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    """
    这个函数是计算relu的导数
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    '''
    copy：BOOL，可选
如果为true（默认值），那么对象被复制。否则，副本将仅当__array__返回副本，如果obj是一个嵌套序列，或者做出是否需要拷贝，以满足任何其他要求（DTYPE，订单等）
    '''
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0  # numpy的语句相当简洁，不用使用if判断。
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    这个函数是计算sigmoid函数的导数
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z)) 
    dZ = dA * s * (1-s)    # 这里就看到了，激活函数的求导不像 前面线性计算  线性的就是矩阵运算，激活函数的就是对应元素相乘，即数组运算
    
    assert (dZ.shape == Z.shape)
    
    return dZ

