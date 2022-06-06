# -*- coding: utf-8 -*-
"""
# @file name  : reorder_flower_dataset.py
# @brief      : 将flower数据集按类别排放，便于分析
"""
import os
import shutil

def my_mkdir(my_dir):
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)


if __name__ == '__main__':
    root_dir = r"../../data/classification_dataset/flowers102"
    path_mat = r"../../data/classification_dataset/flowers102/imagelabels.mat"
    reorder_dir = os.path.join(root_dir, "reorder")
    jpg_dir = os.path.join(root_dir, "jpg") # path of original data

    from scipy.io import loadmat
    label_array = loadmat(path_mat)["labels"].squeeze()

    names = os.listdir(jpg_dir) 
    names = [p for p in names if p.endswith(".jpg")] # get all the jpg images
    for name in names:
        idx = int(name[6:11])
        label = label_array[idx-1]-1 # label start with 0
        out_dir = os.path.join(reorder_dir, str(label)) # 文件夹路径
        path_src = os.path.join(jpg_dir, name) # 具体文件路径
        my_mkdir(out_dir)
        shutil.copy(path_src, out_dir)      # 复制文件