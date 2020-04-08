import spectral
import matplotlib.pyplot  as plt
from sklearn.svm import SVC
import numpy as np
from scipy.io import loadmat
import gdal
from numba import jit
import pandas as pd
import joblib
from sklearn.model_selection import KFold , train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import spectral


#读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName+"文件无法打开")
    return dataset
print("提取数据中......")
dataset = readTif(r"灵兴镇影像/灵兴1.tif")
dataset_label=readTif(r"灵兴镇影像/shp转栅格/栅格合并3.tif")
im_label_width=dataset_label.RasterXSize
im_label_height=dataset_label.RasterYSize
im_width = dataset.RasterXSize #栅格矩阵的列数
im_height = dataset.RasterYSize #栅格矩阵的行数
import_image = dataset.ReadAsArray(0,0,im_width,im_height)#获取数据
output_image=dataset_label.ReadAsArray(0,0,im_label_width,im_label_height)

# mat文件的导入
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import spectral

testdata = np.genfromtxt('C:/Users/asus/Desktop/毕业设计/Code/test/KSC.csv',delimiter=',')
data_test = testdata[:,:-1]
label_test = testdata[:,-1]

clf = joblib.load("KSC_MODEL.m")

predict_label = clf.predict(data_test)
accuracy = metrics.accuracy_score(label_test, predict_label)*100

print(accuracy) # 97.1022836308

output_image=dataset_label.ReadAsArray(0,0,im_label_width,im_label_height)
output_image=output_image.swapaxes(0,1)
import_image=import_image.swapaxes(0,2)

# 将预测的结果匹配到图像中
new_show = np.zeros((output_image.shape[0],output_image.shape[1]))
k = 0
for i in range(output_image.shape[0]):
    for j in range(output_image.shape[1]):
        if output_image[i][j] != 0 :
            new_show[i][j] = predict_label[k]
            k +=1

new_datawithlabel_list = []
for i in range (output_image.shape[0]):
    for j in range (output_image.shape[1]):
        c2l = list (import_image[i][j])
        new_datawithlabel_list.append(c2l)

new_datawithlabel_array = np.array(new_datawithlabel_list)#给import_image,添加相应标签
#标准化数据并储存
from sklearn import preprocessing
data_D = preprocessing.StandardScaler().fit_transform(new_datawithlabel_array)#标准化data
predict_label2 = clf.predict(data_D)
new_show2 = np.zeros((output_image.shape[0],output_image.shape[1]))
k = 0
for i in range(output_image.shape[0]):
    for j in range(output_image.shape[1]):
        new_show2[i][j] = predict_label2[k]
        k +=1
# 展示地物
output_image=output_image.swapaxes(0,1)
new_show=new_show.swapaxes(0,1)
new_show2=new_show2.swapaxes(0,1)
ground_truth = spectral.imshow(classes = output_image.astype(int),figsize =(9,9))
ground_predict = spectral.imshow(classes = new_show.astype(int), figsize =(9,9))
ground_test=spectral.imshow(classes = new_show2.astype(int), figsize =(9,9))
plt.show(ground_truth)
plt.show(ground_predict)
plt.show(ground_test)

print( 'Done')