#!/usr/bin/env python3
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from os import listdir
from os.path import isfile, join

def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    plt.suptitle("Center slices for EPI image") 

def show_one(slice):
    plt.imshow(slice, cmap="gray", origin="lower")
    plt.suptitle("Center slices for EPI image")  

def read_img(filepath):
    img = nib.load(filepath)
    header = img.header
    epi_img_data = img.get_fdata()
    epi_img_data = np.moveaxis(epi_img_data, 2, 0)
    epi_img_data = np.expand_dims(epi_img_data, -1)
    return epi_img_data, header

def read_all(rootpath, path, filname):
    imgs = []
    for d in listdir(rootpath):
        if not isfile(join(rootpath, d)):
            print(rootpath + d + path + filname)
            img, hdr = read_img(rootpath + d + path + filname)
            imgs.append(img)
    return imgs


