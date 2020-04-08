#添加一个计时器
import time
start = time.time()

import os, shutil, glob
from osgeo import gdal

# 定义一个镶嵌的函数，传入的参数是需要镶嵌的数据的列表，以及输出路径
def mosaic(data_list, out_path):

    # 读取其中一个栅格数据来确定镶嵌图像的一些属性
    o_ds = gdal.Open(data_list[0])
    # 投影
    Projection = o_ds.GetProjection()
    # 波段数据类型
    o_ds_array = o_ds.ReadAsArray()

    if 'int8' in o_ds_array.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in o_ds_array.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 像元大小
    transform = o_ds.GetGeoTransform()
    pixelWidth = transform[1]
    pixelHeight = transform[5]

    del o_ds

    minx_list = []
    maxX_list = []
    minY_list = []
    maxY_list = []

    # 对于每一个需要镶嵌的数据，读取它的角点坐标
    for data in data_list:

        # 读取数据
        ds = gdal.Open(data)
        rows = ds.RasterYSize
        cols = ds.RasterXSize

        # 获取角点坐标
        transform = ds.GetGeoTransform()
        minX = transform[0]
        maxY = transform[3]
        pixelWidth = transform[1]
        pixelHeight = transform[5]  # 注意pixelHeight是负值
        maxX = minX + (cols * pixelWidth)
        minY = maxY + (rows * pixelHeight)

        minx_list.append(minX)
        maxX_list.append(maxX)
        minY_list.append(minY)
        maxY_list.append(maxY)

        del ds

    # 获取输出图像坐标
    minX = min(minx_list)
    maxX = max(maxX_list)
    minY = min(minY_list)
    maxY = max(maxY_list)

    # 获取输出图像的行与列
    cols = int((maxX - minX) / pixelWidth)
    rows = int((maxY - minY) / abs(pixelHeight))# 注意pixelHeight是负值

    # 计算每个图像的偏移值
    xOffset_list = []
    yOffset_list = []
    i = 0

    for data in data_list:
        xOffset = int((minx_list[i] - minX) / pixelWidth)
        yOffset = int((maxY_list[i] - maxY) / pixelHeight)
        xOffset_list.append(xOffset)
        yOffset_list.append(yOffset)
        i += 1

    # 创建一个输出图像
    driver = gdal.GetDriverByName("GTiff")
    dsOut = driver.Create(out_path + ".tif", cols, rows, 1, datatype)
    bandOut = dsOut.GetRasterBand(1)

    i = 0
    #将原始图像写入新创建的图像
    for data in data_list:
        # 读取数据
        ds = gdal.Open(data)
        data_band = ds.GetRasterBand(1)
        data_rows = ds.RasterYSize
        data_cols = ds.RasterXSize

        data = data_band.ReadAsArray(0, 0, data_cols, data_rows)
        bandOut.WriteArray(data, xOffset_list[i], yOffset_list[i])

        del ds
        i += 1

    # 设置输出图像的几何信息和投影信息
    geotransform = [minX, pixelWidth, 0, maxY, 0, pixelHeight]
    dsOut.SetGeoTransform(geotransform)
    dsOut.SetProjection(Projection)

    del dsOut

def main():

    input_folder = "灵兴镇影像\\shp转栅格"
    file_list = glob.glob(input_folder + "\\*")

    out_file = "D:\\cnblogs\\data\\china_moasic"
    if os.path.exists(out_file):
        shutil.rmtree(out_file)
        os.mkdir(out_file)
    else:
        os.mkdir(out_file)

    for file in file_list:
        basename = os.path.basename(file)
        out_path = out_file + "\\" + basename

        data_list = glob.glob(file + "\\" + "*.tif")
        print(data_list)

        try:
            mosaic(data_list, out_path)
            print(file + "镶嵌结束")
        except:
            bad_list.append(file)
            print(file + "数据超过4G或其他原因导致无法镶嵌")

bad_list = []
main()
print("无法镶嵌的文件包括如下")
print (bad_list)

end = time.time()
print ("程序运行时间{:.2f}分钟".format((end-start)/60.0))
