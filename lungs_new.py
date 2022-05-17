import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp 
# import gui
import cv2
import matplotlib.image as mpimg

# from mayavi import mlab
from scipy import signal
# from myshow import myshow, myshow3d
from read_data import LoadData, main
from lung_segment import LungSegment
from vessel_segment import VesselSegment
from mpl_toolkits.mplot3d import Axes3D

# 7 10 11 12 14 18 19 20
# 6 8 9 13
# loading data
# data_path = "/hdd/2/wyn/ct-intensity-segmentation/"
# img_name = "VESSEL12_02.mhd"
def process(data_path, img_name):
    num = img_name.split('_')[1].split('.')[0]
    data = LoadData(data_path, img_name)
    data.loaddata()
    print ("the shape of image is ", data.image.GetSize())
    print(num)


    WINDOW_LEVEL = (1050,500)
    ls = LungSegment(data.image)

    # 将图像转换为uint8以供显示
    ls.conv_2_uint8(WINDOW_LEVEL)
    ls.image_showing('ToUint8')

    # 手动设置种子点...
    seed_pts = [(187,245,257), (325,238,257)]

    # 计算区域增长
    ls.regiongrowing(seed_pts)
    ls.image_showing('seed_pts')

    # 显示图片
    ls.image_showing("Region Growing Result")

    # sitk.WriteImage(ls.temp_img, "seg_implicit_thresholds.mhd")

    # 形态运算(闭合)
    ls.image_closing(7)

    # 写入图像
    sitk.WriteImage(ls.temp_img, "img_closing.mhd")

    img_closing = sitk.ReadImage("img_closing.mhd") # reading the existed closing image 

    # 获取3D闭幕图像的Numpy数组以供将来使用
    img_closing_ndarray = sitk.GetArrayFromImage(img_closing)

    # 得到先前肺分割的结果。
    img_closing_ndarray = sitk.GetImageFromArray(img_closing_ndarray)

    vs = VesselSegment(original=data.image, closing=img_closing_ndarray)

    print("   Pricessing Generate lung mask...")
    # vs.generate_lung_mask(lunglabel=[1,-5000], offset = 0)
    vs.generate_lung_mask(offset = 0)

    # Write image...
    Lung_mask = sitk.GetImageFromArray(vs.img)
    sitk.WriteImage(Lung_mask, "Lung_mask_"+num+".mhd")

    print("   Processing Downsampling...")
    vs.downsampling()

    print("   Processing Thresholding...")
    vs.thresholding(thval=180)
    down = sitk.GetImageFromArray(vs.temp_img)
    # sitk.WriteImage(down, "downsample.mhd")

    print("   Processing Region Growing...")
    vs.max_filter(filter_size=2)

    # save the vasculature-segmented image
    filtered = sitk.GetImageFromArray(vs.temp_img)
    sitk.WriteImage(filtered, "filtered.mhd")

    # 转换为二进制映像
    filtered = sitk.ReadImage("filtered.mhd")
    filtered = sitk.GetArrayFromImage(filtered)
    filtered[filtered > 0] = 1
    binary_filtered = sitk.GetImageFromArray(filtered)
    sitk.WriteImage(binary_filtered, "binary_filtered.mhd")

    lung_mask = LoadData(path="", name="Lung_mask_"+num+".mhd")
    lung_mask.loaddata()
    # fissure = LoadData(path="fissure_enhancement_cxx/", name="voxel_val_region_growing_closing.mhd")
    # fissure.loaddata()
    vessel = LoadData(path="", name="binary_filtered.mhd")
    vessel.loaddata()

    lung_mask = sitk.GetArrayFromImage(lung_mask.image)
    # fissure = sitk.GetArrayFromImage(fissure.image)
    vessel = sitk.GetArrayFromImage(vessel.image)

    # lung_mask[lung_mask != 0] = 3
    lung_mask[vessel > 0] = 1
    lung_mask[vessel == 0] = 0
    # lung_mask[fissure > 0] = 2
    ls.image_showing('YUZHI')
    lung_mask = sitk.GetImageFromArray(lung_mask)
    sitk.WriteImage(lung_mask, "label_map_"+num+".mhd")

if __name__ == "__main__":
    data_path = 'VESSEL12/'
    data_list = []
    for i in os.listdir(data_path):
        if '.mhd' in i:
            data_list.append(i)
    for mhd_file in data_list:
        process(data_path, mhd_file)


