# -*- coding: utf-8 -*-
import os
import re
import numpy as np
from sklearn import svm
train_path = "灵兴镇影像/测试样本"
test_path = "灵兴镇影像/训练样本"

#按照文件夹名返回标签和图片地址
def imglist2 (path):
    img = []
    label = []
    a=0
    for file in os.listdir(path):
        file_img = os.path.join(path,file)
        if os.path.isdir(file_img):
            img = img+ os.listdir(file_img)
        for i in range(len(os.listdir(file_img))):
          label=label+[a]
        a=a+1
    return img,label

#将文件夹下按文件名分类的图片制作标签数据txt
def generate(path,label):
    files = os.listdir(dir)
    files.sort()
    listText = open(path+'/'+'list.txt','w')
    for file in files:
        imgsl = os.path.join(path,file)
        imgs=os.listdir(imgsl)
        for img in imgs:
            fileType = os.path.split(file)
            if fileType[1] == '.txt':
              continue
            name = img + ' ' + str(int(label)) +'\n'
            listText.write(name)
    listText.close()
#将文件夹下所有图片按命名制作标签txt
def generate2(dir):
    files = os.listdir(dir)
    files.sort()
    listText = open(dir+'/'+'list.txt','w')
    for file in files:
        img_Water=re.search("旱地",file)
        img_Tree=re.search("植被",file)
        img_Building=re.search("建筑",file)

        img_overpass=re.search("overpass",file)#0
        img_airplane=re.search("airplane",file)#1
        img_parkinglot=re.search("parkinglot",file)#2
        img_river=re.search("river",file)#3
        img_runway=re.search("runway",file)#4
        img_sparseresidential=re.search("sparseresidential",file)#5
        img_storagetanks=re.search("storagetanks",file)#6
        img_tenniscourt=re.search("tenniscourt",file)#7
        img_agricultural=re.search("agricultural",file)#8
        img_baseballdiamond=re.search("baseballdiamond",file)#9
        img_beach=re.search("beach",file)#10
        img_buildings=re.search("buildings",file)#11
        img_chaparral=re.search("chaparral",file)#12
        img_denseresidential=re.search("denseresidential",file)#13
        img_forest=re.search("forest",file)#14
        img_freeway=re.search("freeway",file)#15
        img_golfcourse=re.search("golfcourse",file)#16
        img_harbor=re.search("harbor",file)#17
        img_intersection=re.search("intersection",file)#18
        img_mediumresidential=re.search("mediumresidential",file)#19
        img_mobilehomepark=re.search("mobilehomepark",file)#20
        fileType = os.path.split(file)
        if fileType[1] == '.txt':
            continue
        if img_overpass is not None:
            name = file + ' ' + str(int(0)) +'\n'
        elif img_airplane is not None:
            name = file + ' ' + str(int(1)) +'\n'
        elif img_parkinglot is not None:
            name = file + ' ' + str(int(2)) +'\n'
        elif img_river is not None:
            name = file + ' ' + str(int(3)) +'\n'
        elif img_runway is not None:
            name = file + ' ' + str(int(4)) +'\n'
        elif img_sparseresidential is not None:
            name = file + ' ' + str(int(5)) +'\n'
        elif img_storagetanks is not None:
            name = file + ' ' + str(int(6)) +'\n'
        elif img_tenniscourt is not None:
            name = file + ' ' + str(int(7)) +'\n'
        elif img_agricultural is not None:
            name = file + ' ' + str(int(8)) +'\n'
        elif img_baseballdiamond is not None:
            name = file + ' ' + str(int(9)) +'\n'
        elif img_beach is not None:
            name = file + ' ' + str(int(10)) +'\n'
        elif img_buildings is not None:
            name = file + ' ' + str(int(11)) +'\n'
        elif img_chaparral is not None:
            name = file + ' ' + str(int(12)) +'\n'
        elif img_denseresidential is not None:
            name = file + ' ' + str(int(13)) +'\n'
        elif img_forest is not None:
            name = file + ' ' + str(int(14)) +'\n'
        elif img_freeway is not None:
            name = file + ' ' + str(int(15)) +'\n'
        elif img_golfcourse is not None:
            name = file + ' ' + str(int(16)) +'\n'
        elif img_harbor is not None:
            name = file + ' ' + str(int(17)) +'\n'
        elif img_intersection is not None:
            name = file + ' ' + str(int(18)) +'\n'
        elif img_mediumresidential is not None:
            name = file + ' ' + str(int(19)) +'\n'
        elif img_mobilehomepark is not None:
            name = file + ' ' + str(int(20)) +'\n'
        elif img_Water is not None:
            name = file + ' ' + str(int(1)) +'\n'
        elif img_Building is not None:
            name = file + ' ' + str(int(2)) +'\n'
        elif img_Tree is not None:
            name = file + ' ' + str(int(3)) +'\n'
        else:
            name=file+' '+str(int(-9999))+'\n'
        listText.write(name)

    listText.close()

"""
提取glcm纹理特征
"""
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops


def get_inputs(path): # s为图像路径
    input = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # 读取图像，灰度模式

    # 得到共生矩阵，参数：图像矩阵，距离，方向，灰度级别，是否对称，是否标准化
    glcm = greycomatrix(
        input, [
            2, 8, 16], [
            0, np.pi / 4, np.pi / 2, np.pi * 3 / 4], 256, symmetric=True, normed=True)

    print(glcm)

    #得到共生矩阵统计值，http://tonysyu.github.io/scikit-image/api/skimage.feature.html#skimage.feature.greycoprops
    for prop in {'contrast', 'dissimilarity',
                 'homogeneity', 'energy', 'correlation', 'ASM'}:
        temp = greycoprops(glcm, prop)
        # temp=np.array(temp).reshape(-1)
        print(prop, temp)

    # plt.imshow(input,cmap="gray")
    # plt.show()



import os
import numpy as np
from osgeo import gdal
import glob


list_tif=glob.glob('D:\data\*.tif')
out_path='D:/'

for tif in list_tif:
    in_ds=gdal.Open(tif)
    # 获取文件所在路径以及不带后缀的文件名
    (filepath,fullname)=os.path.split(tif)
    (prename,suffix)=os.path.splitext(fullname)
    if in_ds is None:
        print('Could not open the file '+tif)
    else:
        # 将MODIS原始数据类型转化为反射率
        red=in_ds.GetRasterBand(1).ReadAsArray()*0.0001
        nir=in_ds.GetRasterBand(2).ReadAsArray()*0.0001
        ndvi=(nir-red)/(nir+red)
        # 将NAN转化为0值
        nan_index=np.isnan(ndvi)
        ndvi[nan_index]=0
        ndvi=ndvi.astype(np.float32)
        # 将计算好的NDVI保存为GeoTiff文件
        gtiff_driver=gdal.GetDriverByName('GTiff')
        # 批量处理需要注意文件名是变量，这里截取对应原始文件的不带后缀的文件名
        out_ds=gtiff_driver.Create(out_path+prename+'_ndvi.tif',ndvi.shape[1],ndvi.shape[0],1,gdal.GDT_Float32)
        # 将NDVI数据坐标投影设置为原始坐标投影
        out_ds.SetProjection(in_ds.GetProjection())
        out_ds.SetGeoTransform(in_ds.GetGeoTransform())
        out_band=out_ds.GetRasterBand(1)
        out_band.WriteArray(ndvi)
        out_band.FlushCache()
if __name__ == '__main__':
    #img_all,label_all=imglist2(train_path)
    generate2(test_path)
    #generate2(train_path)
 
