import dataProcess as dp
import numpy as np
import scipy
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
from LogisticModel import LogisticModel

if __name__ == '__main__':
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = dp.load_dataset()
    train_set_flat = dp.processing(train_set_x_orig)
    test_set_flat = dp.processing(test_set_x_orig)
    lm = LogisticModel()
    result = lm.model(train_set_flat,train_set_y_orig,test_set_flat,test_set_y_orig,2000,0.005)


