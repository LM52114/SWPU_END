import spectral
import matplotlib.pyplot  as plt
from sklearn.svm import SVC
import numpy as np
from scipy.io import loadmat
'''
get the KSC data
'''
import_image = loadmat(r'C:/Users/asus/Desktop/毕业设计/Code/test/KSC.mat')['paviaU']
output_image = loadmat(r'C:/Users/asus/Desktop/毕业设计/Code/test/KSC_gt.mat')['paviaU_gt']

#input_image.shape#:(610, 340, 103)长610，高340，波段数103，每一处的值代表像元值
# # output_image.shape#:(610, 340)每一处的值代表类型编号
np.unique(output_image)#去除重复元素，并排序输出

'''
get the number of each class

'''
#统计每类样本个数
dict_k = {}
for i in range(output_image.shape[0]):
    for j in range(output_image.shape[1]):
        if output_image[i][j] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            if output_image[i][j] not in dict_k:
                dict_k[output_image[i][j]] = 0
            dict_k[output_image[i][j]] += 1
# print dict_k   #{1: 761, 2: 243, 3: 256, 4: 252, 5: 161, 6: 229, 7: 105, 8: 431, 9: 520, 10: 404, 11: 419, 12: 503, 13: 927}
# print reduce(lambda x,y:x+y, dict_k.values())
#add up all of the valuse in dictionary of dict_k

'''
show the picture of HSI
'''
# ground_truth = spectral.imshow(classes=output_image.astype(int), figsize=(5,5))
# ksc_color = np.array()
# ground_truth = spectral.imshow(classes= output_image.astype(int), figsize=(6,6))
# plt.show(ground_truth)    #  it shows the original picture


#if u want the picture shows differents colors , do next

# ksc_color = np.array([
#     [255,255,255],
#     [184,40,99],
#     [74,77,145],
#     [35,102,193],
#     [238,110,105],
#     [117,249,76],
#     [114,251,253],
#     [126,196,59],
#     [234,65,247],
#     [141,79,77],
#     [183,40,99],
#     [0,39,245],
#     [90,196,111],
# ])
# ground_truth = spectral.imshow(classes= output_image.astype(int), figsize=(9,9),colors = ksc_color)
# plt.show(ground_truth)

'''
change mat to csv
'''
#重构需要用到的类

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

#标准化数据并储存
from sklearn import preprocessing
data_D = preprocessing.StandardScaler().fit_transform(new_datawithlabel_array[:,:-1])#标准化data
data_L = new_datawithlabel_array[:,-1]#拿到lable

import pandas as pd
new = np.column_stack((data_D,data_L))#将两个矩阵按列合并
new_ = pd.DataFrame(new)#将new转换为表的形式
new_.to_csv(r'C:/Users/asus/Desktop/毕业设计/Code/test/KSC.csv',header = False , index = False)#存储为csv文件
#the above get the csv data

'''
Train the model , save the model

'''
import joblib
from sklearn.model_selection import KFold , train_test_split
from sklearn import metrics

#split train and test data
data = pd.read_csv('C:/Users/asus/Desktop/毕业设计/Code/test/KSC.csv', header= None)
data = data.values
data_D = data [:,:-1]
data_L = data[:,-1]
data_train, data_test, label_train, label_test = train_test_split(data_D,data_L,test_size= 0.5)

#train the model
clf = SVC(kernel= 'rbf', gamma = 0.125, C= 16)
clf.fit(data_train,label_train)
pred = clf.predict(data_test)
accuracy = metrics.accuracy_score(label_test,pred)*100
print(accuracy)

#储存学习模型
joblib.dump(clf,'KSC_MODEL.m')


'''

模型预测，在图中标记

'''
#

# mat文件的导入
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import spectral


# KSC
input_image = loadmat('C:/Users/asus/Desktop/毕业设计/Code/test/KSC.mat')['paviaU']
output_image = loadmat('C:/Users/asus/Desktop/毕业设计/Code/test/KSC_gt.mat')['paviaU_gt']


testdata = np.genfromtxt('C:/Users/asus/Desktop/毕业设计/Code/test/KSC.csv',delimiter=',')
data_test = testdata[:,:-1]
label_test = testdata[:,-1]

# /Users/mrlevo/Desktop/CBD_HC_MCLU_MODEL.m
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

# print new_show.shape

# 展示地物

ground_truth = spectral.imshow(classes = output_image.astype(int),figsize =(9,9))
ground_predict = spectral.imshow(classes = new_show.astype(int), figsize =(9,9))
plt.show(ground_truth)
plt.show(ground_predict)

print( 'Done')