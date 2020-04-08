# mat文件的导入
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import spectral
import joblib
import metrics

# KSC
input_image = loadmat('C:/Users/asus/Desktop/毕业设计/Code/test/KSC.mat')['KSC']
output_image = loadmat('C:/Users/asus/Desktop/毕业设计/Code/test/KSC_gt.mat')['KSC_gt']


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

