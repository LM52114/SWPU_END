#coding=utf-8
from PIL import Image
import os
image_width = 100
image_height = 128
# 修改图片的大小
def fixed_size(filePath,savePath):
    """按照固定尺寸处理图片"""
    im = Image.open(filePath)
    out = im.resize((image_width, image_height), Image.ANTIALIAS)
    out.save(savePath)

def changeSize(filePath,destPath):
    if not os.path.exists(destPath):
        os.makedirs(destPath)
    for root, dirs, files in os.walk(filePath):
        for file in files:
            if file[-1]=='g':
                fixed_size(os.path.join(filePath, file), os.path.join(destPath, file))

if __name__ == '__main__':
    filePath1 = r'女神级别'
    destPath1 = r'女神级别格式化'
    filePath2 = r'大众级别'
    destPath2 = r'大众级别格式化'
    filePath3 = r'普通级别'
    destPath3 = r'普通级别格式化'
    changeSize(filePath1,destPath1)
    changeSize(filePath2, destPath2)
    changeSize(filePath3, destPath3)
    print("图片格式化完成")

