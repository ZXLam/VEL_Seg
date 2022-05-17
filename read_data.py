"""
这个文件是读取图片并转换成numpy数组
"""
import SimpleITK as sitk
import matplotlib.pyplot as plt

class LoadData:
    """
    这个类被设计读取一个图片
    """
    def __init__(self, path, name):
        """
        :param path: 图片的路径
        :param name: 图片的名字
        """
        self.img_path = path + name
        self.image = None
        self.slices = None

    def loaddata(self):
        """
        从给定的位置读取图片
        :return: None
        """
        self.image = sitk.ReadImage(self.img_path)

    def tileimage(self, index1, index2):
        """
        将3D图像平铺成两个选定的切片以供显示。
        :param index1: 选定的切片1 slice 1
        :param index2: 选定的切片2 slice 2
        :return: None
        """
        self.slices = sitk.Tile(self.image[:, :, index1],
                                self.image[:, :, index2],
                                (2, 1, 0))

    def sitk_show(self, title=None, margin=0.0, dpi=40):
        """
        显示平铺的2D图像
        :param title: 标题
        :param margin: 页面空白
        :param dpi: 图像的分辨率
        :return: None
        """
        # 把图片的切片转化成numpy的ndarray
        nda = sitk.GetArrayFromImage(self.slices)
        # 图片的大小
        figsize = (1 + margin) * nda.shape[0] / dpi, (1 + margin) * nda.shape[1] / dpi
        extent = (0, nda.shape[1], nda.shape[0], 0)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

        plt.set_cmap("gray")
        ax.imshow(nda, extent=extent, interpolation=None)
        if title:
            plt.title(title)
        plt.show()


def main():
    data_path = "VESSEL12/"
    img_name = "VESSEL12_01.mhd"

    # 要使用‘sitk_show’可视化的切片索引
    idxSlice1 = 26
    idxSlice2 = 50

    # 要分配给分割的灰度图的int标签
    labelGrayMatter = 1

    data = LoadData(data_path, img_name)
    data.loaddata()
    print("after read image...")
    data.tileimage(idxSlice1, idxSlice2)
    data.sitk_show()

    print("after showing image...")
    image_array = sitk.GetArrayFromImage(data.image)
    print("the shape of image is ", image_array.shape)

    # output = sitk.DiscreteGaussianFilter(input, 1.0, 5)
    # sitk.Show(image)

if __name__ == "__main__":
    main()