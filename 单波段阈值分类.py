from osgeo import gdal_array

# 输入文件
src = "D:\学习资料\LC81290392018156LGN00\LC08_L1TP_129039_20180605_20180615_01_T1_B1.TIF"
# 输出文件名
tgt = "classified.jpg"

# 使用gdal库加载图片到numpy
srcArr = gdal_array.LoadFile(src)
# 根据类别数目将直方图分割成20个颜色区间
classes = gdal_array.numpy.histogram(srcArr, bins=20)[1]

# 颜色查找表的记录数必须是len(classes)+1，声明RGN元组
lut = [[255, 0, 0], [191, 48, 48], [166, 0, 0], [255, 64, 64], [255, 155, 155],
       [255, 116, 0], [191, 113, 48], [255, 178, 115], [0, 153, 153], [29, 115, 155],
       [0, 99, 99], [166, 75, 0], [0, 204, 0], [51, 204, 204], [255, 150, 64],
       [92, 204, 204], [38, 153, 38], [0, 133, 0], [57, 230, 57], [103, 230, 103],
       [184, 138, 0]]

# 分类初始值
start = 1

# 创建一个RGB颜色的JPEG图片输出
rgb = gdal_array.numpy.zeros((3, srcArr.shape[0], srcArr.shape[1], ), gdal_array.numpy.float32)

# 处理所有类并声明颜色
for i in range(len(classes)):
    mask = gdal_array.numpy.logical_and(start <= srcArr, srcArr <= classes[i])
    for j in range(len(lut[i])):
        rgb[j] = gdal_array.numpy.choose(mask, (rgb[j], lut[i][j]))
    start = classes[i] + 1

# 保存图片
output = gdal_array.SaveArray(rgb.astype(gdal_array.numpy.uint8), tgt, format="JPEG")
output = None