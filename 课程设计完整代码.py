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
import spectral

#读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName+"文件无法打开")
    return dataset
print("提取数据中......")
dataset = readTif(r"灵兴镇影像/灵兴1.tif")
dataset_label=readTif(r"灵兴镇影像/shp转栅格/栅格合并4.tif")
im_label_width=dataset_label.RasterXSize
im_label_height=dataset_label.RasterYSize
im_width = dataset.RasterXSize #栅格矩阵的列数
im_height = dataset.RasterYSize #栅格矩阵的行数
im_geotrans = dataset.GetGeoTransform()#获取仿射矩阵信息
im_proj = dataset.GetProjection()#获取投影信息
import_image = dataset.ReadAsArray(0,0,im_width,im_height)#获取数据
output_image=dataset_label.ReadAsArray(0,0,im_label_width,im_label_height)
import_image=import_image.swapaxes(0,2)
output_image=output_image.swapaxes(0,1)
np.unique(output_image)#去除重复元素，并排序输出
print(output_image)

#统计每类样本个数
dict_k = {}
for i in range(output_image.shape[0]):
    for j in range(output_image.shape[1]):
        if output_image[i][j] in [1, 2, 3, 4]:
            if output_image[i][j] not in dict_k:
                dict_k[output_image[i][j]] = 0
            dict_k[output_image[i][j]] += 1
#重构需要用到的类
print("数据提取完毕，开始为数据添加标签......")
need_label = np.zeros([output_image.shape[0],output_image.shape[1]])#用0.填充一个output_image.shape[0]行,output_image.shape[1]列的array
new_datawithlabel_list = []
for i in range(output_image.shape[0]):#去除output_image为0的元素
    for j in range (output_image.shape[1]):
        if output_image[i][j] != 0 :
            need_label[i][j]=output_image[i][j]

for i in range (output_image.shape[0]):
    for j in range (output_image.shape[1]):
            if need_label[i][j] != 0 :
                c2l = list (import_image[i][j])
                c2l.append (need_label[i][j])
                new_datawithlabel_list.append(c2l)

new_datawithlabel_array = np.array(new_datawithlabel_list)#给import_image,添加相应标签
print("标签制作完成，将训练数据制作为csv表格......")
#标准化数据并储存
from sklearn import preprocessing
data_D = preprocessing.StandardScaler().fit_transform(new_datawithlabel_array[:,:-1])#标准化data
data_L = new_datawithlabel_array[:,-1]#拿到lable


new = np.column_stack((data_D,data_L))#将两个矩阵按列合并
new_ = pd.DataFrame(new)#将new转换为表的形式
new_.to_csv(r'C:/Users/asus/Desktop/毕业设计/Code/test/KSC.csv',header = False , index = False)#存储为csv文件

print("表格已经存储，开始训练模型......")


#split train and test data
data = pd.read_csv('C:/Users/asus/Desktop/毕业设计/Code/test/KSC.csv', header= None)
data = data.values
data_D = data [:,:-1]
data_L = data[:,-1]
data_train, data_test, label_train, label_test = train_test_split(data_D,data_L,test_size= 0.7)

#train the model
clf = SVC(kernel= "rbf", gamma = 0.125,cache_size=8000, C= 16)
clf.fit(data_train,label_train)
pred = clf.predict(data_test)
accuracy = metrics.accuracy_score(label_test,pred)*100
print(accuracy)

#储存学习模型
joblib.dump(clf,'KSC_MODEL.m')

print("模型保存结束，开始预测......")

# mat文件的导入


testdata = np.genfromtxt('C:/Users/asus/Desktop/毕业设计/Code/test/KSC.csv',delimiter=',')
data_test = testdata[:,:-1]
label_test = testdata[:,-1]

clf = joblib.load("KSC_MODEL.m")

predict_label = clf.predict(data_test)
accuracy = metrics.accuracy_score(label_test, predict_label)*100

print(accuracy) # 97.1022836308


# 将预测的结果匹配到图像中
new_show = np.zeros((output_image.shape[0],output_image.shape[1]))
k = 0
for i in range(output_image.shape[0]):
    for j in range(output_image.shape[1]):
        if output_image[i][j] != 0 :
            new_show[i][j] = predict_label[k]
            k +=1

# 展示地物

ground_truth = spectral.imshow(classes = output_image.astype(int),figsize =(9,9))
ground_predict = spectral.imshow(classes = new_show.astype(int), figsize =(9,9))
plt.show(ground_truth)
plt.show(ground_predict)

print( 'Done')