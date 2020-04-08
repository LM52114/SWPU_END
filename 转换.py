import cv2
import numpy as np
import h5py
import math
import glob
import os
import scipy.io as io
from PIL import Image
from libtiff import TIFF

def save_to_mat(img,output_name):
    new_data_path = os.path.join(os.getcwd(),"matType")
    if not os.path.isdir(new_data_path):
        os.mkdir(new_data_path)
    npy_data = np.array(img,dtype= "uint16")
    np.save(new_data_path+'/{}.npy'.format(output_name),npy_data)
    npy_load = np.load(new_data_path+'/{}.npy'.format(output_name))
    io.savemat(new_data_path+'/{}.mat'.format(output_name),{'data':npy_load})

if __name__ == '__main__':
    imagetif=TIFF.open('handan.tif', mode = "r")
    save_to_mat(imagetif,"handan")