# -*- coding: utf-8 -*-
"""
# @file name  : flower_102.py
# @brief      : read flower 102 dataset
"""

import os
import random
from PIL import Image
from torch.utils.data import Dataset


class FlowerDataset(Dataset):
    cls_num = 102
    names = tuple([i for i in range(cls_num)])

    def __init__(self, root_dir, transform=None):
        """
        获取数据集的路径、预处理的方法
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_info = []   # [(path, label), ... , ]
        self.label_array = None
        self._get_img_info() # run the function to get the information of files

    def __getitem__(self, index):
        """
        输入标量index, 从硬盘中读取数据，并预处理，to Tensor
        :param index:
        :return:
        """
        path_img, label = self.img_info[index] 
        img = Image.open(path_img).convert('RGB') #  Image.open(path_img) get GBR, we need to convert it into RGB

        if self.transform is not None:
            img = self.transform(img)

        return img, label, path_img

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(
                self.root_dir))   # 代码具有友好的提示功能，便于debug
        return len(self.img_info)

    def _get_img_info(self):
        """
        实现数据集的读取，将硬盘中的数据路径，标签读取进来，存在一个list中
        path, label
        :return:
        """
        names_imgs = os.listdir(self.root_dir)
        names_imgs = [n for n in names_imgs if n.endswith(".jpg")] # select all the jog file from root_dir

        # 读取mat形式label
        label_file = "imagelabels.mat"  # hard code
        path_label_file = os.path.join(self.root_dir, "..", label_file)
        from scipy.io import loadmat
        label_array = loadmat(path_label_file)["labels"].squeeze() # delete one dimension
        self.label_array = label_array

        # 匹配label
        idx_imgs = [int(n[6:11]) for n in names_imgs] # convert string to int

        path_imgs = [os.path.join(self.root_dir, n) for n in names_imgs]# get full path for each file
        # get file path and label
        self.img_info = [(p, int(label_array[idx-1]-1)) for p, idx in zip(path_imgs, idx_imgs)]   # 注意索引，注意标签减一


if __name__ == "__main__":
    root_dir = r"../../data/classification_dataset/flowers102/train"

    test_dataset = FlowerDataset(root_dir)
 
    print(len(test_dataset))
    print(next(iter(test_dataset))) # 通过next得到迭代器的第一个元素


