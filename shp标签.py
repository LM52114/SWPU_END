import operator
from osgeo import gdal, gdal_array, osr
import shapefile
from PIL import ImageDraw, Image

# 用于裁剪的栅格影像
raster = "灵兴镇影像/灵兴1.tif"
# 用于裁剪的多边形shapefile文件，注意这里路径的问题
shp = "hancock/hancock"
# 裁剪后的栅格文件名
output = "clip"


def imageToArray(i):
    """
    将一个python影像库的数组转换为一个gdal_array图片
    """
    a = gdal_array.numpy.fromstring(i.tostring(), 'b')
    a.shape = i.im.size[1], i.im.size[0]
    return a


def world2pixel(geoMatrix, x, y):
    """
    使用gdal库的geomatrix对象(gdal.GetGeoTransform())计算地理坐标的像素位置
    """
    ulX = geoMatrix[0]
    ulY = geoMatrix[3]
    xDist = geoMatrix[1]
    yDist = geoMatrix[5]
    rtnX = geoMatrix[2]
    rtnY = geoMatrix[4]
    pixel = int((x - ulX) / xDist)
    line = int((ulY - y) / abs(yDist))
    return (pixel, line)


# 将源数据作为gdal_array载入
srcArray = gdal_array.LoadFile(raster)
# 同时载入gdal库的图片从而获取geotransform
srcImage = gdal.Open(raster)
geoTrans = srcImage.GetGeoTransform()
# 使用pyshp库打开shapefile文件
r = shapefile.Reader("{}.shp".format(shp))
# 将图层扩展转换为图片像素坐标
minX, minY, maxX,maxY = r.bbox
ulX, ulY = world2pixel(geoTrans, minX, maxY)
lrX, lrY = world2pixel(geoTrans, maxX, minY)
# 计算新图像的像素尺寸
pxWidth = int(lrX - ulX)
pxHeight = int(lrY - ulY)
clip = srcArray[:, ulY:lrY, ulX:lrX]
# 为图片创建一个新的geomatrix对象以便附加地理参照数据
geoTrans = list(geoTrans)
geoTrans[0] = minX
geoTrans[3] = maxY
# 在一个空白的8字节黑白遮罩图片上把点映射为像素，绘制县市边界线
pixels = []
for p in r.shape(0).points:
    pixels.append(world2pixel(geoTrans, p[0], p[1]))
rasterPoly = Image.new("L", (pxWidth, pxHeight), 1)
# 使用PIL创建一个空白图片用于绘制多边形
rasterize = ImageDraw.Draw(rasterPoly)
rasterize.polygon(pixels, 0)
# 将PIL图片转换为numpy数组
mask = imageToArray(rasterPoly)
# 根据掩模图层对图片进行裁剪
clip = gdal_array.numpy.choose(mask, (clip, 0)).astype(gdal_array.numpy.uint8)
# 将裁剪图像保存为tiff
gdal_array.SaveArray(clip, "{}.tif".format(output), format="GTIFF", prototype=raster)