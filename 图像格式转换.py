import os
from osgeo import gdal
 #转png,jpg
open_path = "D:/segmentation/datasets/tif/"
save_path = "D:/segmentation/datasets/png/"

images = os.listdir(open_path)
for image in images:
    im=gdal.Open(os.path.join(open_path,image))
    driver=gdal.GetDriverByName('PNG')
    dst_ds = driver.CreateCopy(os.path.join(save_path,image.split('.')[0]+".png"), im)