# -*- coding:utf-8 -*-

"""
图片任意拼接，参数化形式代码
使用指南：
    1. 修改常量的数值，可以实现不同样子的图片拼接，例如拼接成5*20，或者100*200的大图，每张小图也可以控制大小
    2. 可以自定义函数让图片不仅仅是全部拼接成一张图，也可以自定义哪些图进行拼接。
"""

import PIL.Image as Image
import os

IMAGES_PATH = "D:/学习资料/test/灵兴镇影像/切割/"  # 图片集地址
IMAGES_FORMAT = ['.tif','.TIF']  # 图片格式
IMAGE_SIZE = 16  # 图片大小
IMAGE_ROW = 73  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 112  # 图片间隔，也就是合并成一张图后，一共有几列
IMAGE_SAVE_PATH = 'toImage_0.tif'  # 图片转换后的地址

# 获取图片集地址下的所有图片名称
image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]
print(image_names)
# 简单的对于参数的设定和实际图片集的大小进行数量判断
if len(image_names) != IMAGE_ROW * IMAGE_COLUMN:
    raise ValueError("合成图片的参数和要求的数量不能匹配！")


# 图像拼接
def image_compose():
    # 打开一个新的图
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))
    # 循环遍历，把每张图按顺序粘贴到对应位置上
    for y in range(0, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):

            from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize((IMAGE_SIZE, IMAGE_SIZE),Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
    # 保存新图
    return to_image.save(IMAGE_SAVE_PATH)


image_compose()
"""
# 图片拼接
from PIL import Image
# pil paste可以进行图片拼接
import cv2
import numpy as np
import glob as glob
import os

num=73*112
os.chdir(r"D:/学习资料/test/灵兴镇影像/切割")
img_name=[]
for file_name in glob.glob("*.tif"):
    print(file_name)
    img_name.append(file_name)

# 批量处理图片
img_path = glob.glob("D:/学习资料/test/灵兴镇影像/切割/*tif")
for i in range(int(len(img_path)/num)):
    path = img_path[i*num]
    print(path)
    img_out = cv2.imread(path)

    for j in range(1, num):
        path = img_path[i*num+j]
        img_tmp = cv2.imread(path)

        # 横向
        img_out = np.concatenate((img_out, img_tmp), axis=1)

    # cv2.imshow("IMG", img_out)
    cv2.imwrite("D:/学习资料/test/灵兴镇影像/"+img_name[i*num+j][0:-6]+"_out.jpg", img_out)"""

