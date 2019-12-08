import numpy as np
import h5py
import matplotlib.pyplot as plt


'''
LoadDataset()
'''
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes






# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

#查看数据集
print(train_set_x_orig.shape) #(209, 64, 64, 3)
print(train_set_y.shape) #(1, 209)
print(test_set_x_orig.shape)  # (50, 64, 64, 3)
print(test_set_y.shape) #  (1, 50)
print(classes.shape) # (2,)
print(classes) #以二进制形式存储的 [b'non-cat' b'cat']
print(train_set_y)
print(train_set_y[:,1].shape)  #squeeze函数就是为了获得这个0 (1,)
print(classes[np.squeeze(train_set_y[:,1])].decode('utf-8')) #np.squeeze这个函数的作用是 从数组的形状中删除单维度条目，
                                                # 即把shape中为1的维度去掉
plt.imshow(train_set_x_orig[5])
plt.show()





