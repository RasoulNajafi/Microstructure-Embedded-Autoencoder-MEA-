import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage import data
import time 


def split_data(Data, resolution):
    lst = list()
    s_num  = Data.shape[0]
    for i in range(s_num):
        X = Data[i, :] 
        X = X.reshape(resolution , resolution)
        X = np.expand_dims(X, axis=0)
        lst.append(X)
    return lst
    
FE_LR = np.loadtxt(r"K:\Najafi\codes\UNet_SH\test\output_FE_test_LR.txt")
FE_LR = split_data(FE_LR,11)
upsample_factor = 9.2 #9.2 #4.6
ORDER = 1
results=list()
for i in range(0):    
    upsampled_temp = ndimage.zoom(FE_LR[i], upsample_factor, order=ORDER , mode='nearest')
    results.append(upsampled_temp[0])


