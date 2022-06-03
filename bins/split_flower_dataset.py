# -*- coding: utf-8 -*-
"""
# @file name  : split_flower_dataset.py
# @brief      : split flower dataset into train、valid、test set
"""
import os
import pickle
import shutil
import random


def my_mkdir(my_dir):
    '''
    Check if a folder exists, if not, create it.
    ----------
    Parameters
    ----------
    my_dir : path of folder

    Returns
    -------
    None.

    '''
    if not os.path.isdir(my_dir):
        os.makedirs(my_dir)


def move_img(imgs, root_dir, setname):
    '''
    Move images to corresponding sub-folder
    ----------
    Parameters
    ----------
    imgs : TYPE
        DESCRIPTION.
    root_dir : TYPE
        DESCRIPTION.
    setname : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    data_dir = os.path.join(root_dir, setname)
    my_mkdir(data_dir)
    for path_img in imgs:
        print(path_img)
        shutil.copy(path_img, data_dir) # copy image to another folder
    print("{} dataset, copy {} imgs to {}".format(setname, len(imgs), data_dir))


if __name__ == '__main__':
    # 0. config
    random_seed = 20210309
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1
    # 1. read list and shuffle
    root_dir = r"../../data/classification_dataset/flowers102"
    data_dir = os.path.join(root_dir, "jpg")
    name_imgs = [p for p in os.listdir(data_dir) if p.endswith(".jpg")]
    path_imgs = [os.path.join(data_dir, name) for name in name_imgs] #join root path and file path
    random.seed(random_seed)
    random.shuffle(path_imgs)
    print(path_imgs[0])

    # 2. random split the file
    train_breakpoints = int(len(path_imgs)*train_ratio)
    valid_breakpoints = int(len(path_imgs)*(train_ratio + valid_ratio))
    train_imgs = path_imgs[:train_breakpoints]
    valid_imgs = path_imgs[train_breakpoints:valid_breakpoints]
    test_imgs = path_imgs[valid_breakpoints:]

    # 3. 复制
    move_img(train_imgs, root_dir, "train")
    move_img(valid_imgs, root_dir, "valid")
    move_img(test_imgs, root_dir, "test")


r"""
train dataset, copy 6551 imgs to \102flowers\train
valid dataset, copy 819 imgs to \102flowers\valid
test dataset, copy 819 imgs to \102flowers\test
"""
