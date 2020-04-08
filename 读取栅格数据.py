import gdal
from numba import jit
import numpy as np
import pandas as pd
import spectral
import matplotlib.pyplot  as plt

#读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName+"文件无法打开")
    return dataset

#保存tif文件函数
def writeTiff(im_data,im_geotrans,im_proj,path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    print(im_bands,im_width,im_height)
    print(type(im_bands),type(im_width),type(im_height))
    #创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans) #写入仿射变换参数
        dataset.SetProjection(im_proj) #写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i+1).WriteArray(im_data[i])
    del dataset

dataset = readTif(r"灵兴镇影像/灵兴1.tif")
dataset_label=readTif(r"灵兴镇影像/shp转栅格/栅格合并.tif")
im_label_width=dataset_label.RasterXSize
im_label_height=dataset_label.RasterYSize
im_width = dataset.RasterXSize #栅格矩阵的列数
im_height = dataset.RasterYSize #栅格矩阵的行数
im_geotrans = dataset.GetGeoTransform()#获取仿射矩阵信息
im_proj = dataset.GetProjection()#获取投影信息
im_data = dataset.ReadAsArray(0,0,im_width,im_height)#获取数据
im_label_data=dataset_label.ReadAsArray(0,0,im_label_width,im_label_height)
im_label_data[:,[0, -1]] = im_label_data[:,[-1, 0]]
ground_predict = spectral.imshow(classes = im_label_data.astype(int), figsize =(9,9))

print(im_data.shape)
print(im_data)
print(im_label_data)
print(im_label_data.shape)

"""im_data=im_data[:,0]
np.delete(im_data,0,axis=1)"""
from sklearn import preprocessing
data_D = preprocessing.StandardScaler().fit_transform(im_label_data[:,:-1])#标准化data
data_L = im_data[:,-1]#拿到lable


new = np.column_stack((data_D,data_L))#将两个矩阵按列合并
new_ = pd.DataFrame(new)#将new转换为表的形式
new_.to_csv(r'C:/Users/asus/Desktop/毕业设计/Code/test/灵兴镇影像/KSC.csv',header = False , index = False)#存储为csv文件
im_data_BGR = im_data[0:3,:,]

@jit
def to8bit(data):
    max_value = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                if(data[i][j][k]>max_value):
                    max_value = data[i][j][k]
    data_8bit = np.zeros(data.shape,np.uint8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                data_8bit[i][j][k] = int(data[i][j][k]*255/max_value)
    return data_8bit

#data_8bit = to8bit(im_data_BGR)

#writeTiff(data_8bit,im_geotrans,im_proj,r"I:\gf2\8bitData\yushu_8bit.tif")