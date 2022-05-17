"""
这个脚本是将肺从背景中分割出来，作为肺血管分割的前置步骤。
"""
import SimpleITK as sitk
# import gui

class LungSegment:
    """
    这个类是为肺部的3D分割而设计的，包括以下方法：
    ...
    """
    def __init__(self, img):
        self.img = img
        self.temp_img = None
        self.img_uint8 = None

    def conv_2_uint8(self, WINDOW_LEVEL=(1050,500)):
        """
        将原始图像转换为8位图像
        :param WINDOW_LEVEL: 使用外部查看器(ITK-SNAP或3DSlicer)，我们确定了视觉上吸引人的窗位设置
        :return: None
        """
        # self.img_uint8 = sitk.Cast(self.img,
        #                           sitk.sitkUInt8)
        self.img_uint8 = sitk.Cast(sitk.IntensityWindowing(self.img,
                                 windowMinimum=WINDOW_LEVEL[1] - WINDOW_LEVEL[0] / 2.0,
                                 windowMaximum=WINDOW_LEVEL[1] + WINDOW_LEVEL[0] / 2.0),
                                 sitk.sitkUInt8)


    def regiongrowing(self, seed_pts):
        """
        通过SimpleITK工具实现对给定种子点的信心连接
        :param seed_pts: 区域生长的种子点[(z，y，x)，...]
        :return: None
        """
        self.temp_img = sitk.ConfidenceConnected(self.img, seedList=seed_pts,
                                                           numberOfIterations=0,
                                                           multiplier=2,
                                                           initialNeighborhoodRadius=1,
                                                           replaceValue=1)

    def image_showing(self, title=''):
        """
        Showing image.
        :return: None
        """
        # gui.MultiImageDisplay(image_list=[sitk.LabelOverlay(self.img_uint8, self.temp_img)],
        #                       title_list=[title])
        pass

    def image_closing(self, size=7):
        """
        实现形态闭合，固定图像内部的“洞”。
        :param size: 关闭内核的大小
        :return: None
        """
        closing = sitk.BinaryMorphologicalClosingImageFilter()
        closing.SetForegroundValue(1)
        closing.SetKernelRadius(size)
        self.temp_img = closing.Execute(self.temp_img)