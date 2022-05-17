"""
这个文件是用来增强肺血管的。
"""
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import cv2

from itertools import zip_longest

class VesselSegment:
    """
    本课程专为增强肺血管功能而设计，包括以下方法：
    1.下采样。
    2.阈值分割。
    3.3D区域生长。
    4.过滤小结构。
    """

    def __init__(self, original, closing):
        """
        :param original: ITK格式的原始图像。
        :param closing:  关闭结果图片。
        :param thval:   阈值(默认值：130HU)
        :param filter_vol: 区域增长后去除小结构的过滤值(默认值：2ml)
        """
        self.original_img = original
        self.closing_img = sitk.GetArrayFromImage(closing)
        self.img = None
        self.thval = None
        self.filter_vol = None
        self.temp_img = None

    def erosion(self, lunglabel=[201, 202]):
        """
        Shrik the region of lung...
        """
        temp = np.zeros_like(self.closing_img)
        temp[self.closing_img == lunglabel[0]] = 1
        temp[self.closing_img == lunglabel[1]] = 1
        Erode_filter = sitk.BinaryErodeImageFilter()

        Erode_filter.SetKernelRadius ( 2 ).SetForegroundValue ( 1 )
        # 设置核半径。
        # 设置前景值
        self.closing_img = sitk.GetArrayFromImage( Erode_filter.Execute ( sitk.GetImageFromArray(temp) ) )

    def generate_lung_mask(self, offset = 0):
        """
        生成肺部遮罩
        :return: None
        """
        self.img = sitk.GetArrayFromImage(self.original_img).copy() - offset
        self.img[self.closing_img == 0] = 0

    def downsampling(self):
        """
        对输入图像进行下采样，从[-1024，0]降至[0,255]，以减少内存需求。
                              / max(0, min(254, (Vorig+1024)/4), v 属于肺区
                        Vds =
                              \ 255,                             不是
        :return: None
        """
        temp = (self.img + 1024) / 4
        temp[temp > 254] = 254
        temp[temp < 0] = 0
        self.temp_img = temp

    def thresholding(self, thval=130):
        """
        用给定的阈值对图像进行阈值处理。
        :return: None
        """
        self.thval = thval
        self.temp_img[self.temp_img < thval] = thval

    def max_filter(self, filter_size=5):
        """
        实现最大值过滤。
        :param filter_size: 过滤器大小
        :return: None
        """
        temp = self.temp_img.copy()
        temp[temp >= 254] = 0
        temp[temp <= self.thval] = 0
        self.temp_img = ndimage.filters.maximum_filter(temp, size=filter_size)

    def filtering(self, min_size=10, max_size=5000):
        """
        去除体积小于给定值(2ml)的小结构
        :param min_size:
        :return: None
        """
        self.filter_vol = min_size
        z, y, x = self.temp_img.shape
        count_labels = []
        for i in range(z):
            dist_transform = cv2.distanceTransform(np.uint8(self.temp_img[i, :, :]), cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            # Marker labelling
            ret, markers = cv2.connectedComponents(sure_fg)
            # Add one to all labels so that sure background is not 0, but 1
            markers += 1
            count_labels = np.asarray([x + y for x, y in zip_longest(count_labels,
                                                                      np.bincount(markers.flatten()),
                                                                      fillvalue=0)])
        labels = np.arange(0, len(count_labels))
        labels[count_labels < min_size] = 0
        labels[count_labels > max_size] = 0
        labels = np.asarray(list(set(labels)))
        for label in labels:
            self.temp_img[self.temp_img == label] = 0