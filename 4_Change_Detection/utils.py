from __future__ import division
import os
import numpy as np
import random

def recursive_glob(rootdir=".", suffix=".tif"):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot,filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix) or filename.endswith('.png') or filename.endswith('.npy')
    ]

def normalization(img, mean, std):
    img = (img-mean)/std
    img = ((img - np.min(img)) / (np.max(img)-np.min(img)))*255
    return img

def random_crop(img, mask, width, height):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == mask.shape[0]
    assert img.shape[1] == mask.shape[1]
    x = random.randint(0, img.shape[1] - width)
    y = random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    mask = mask[y:y+height, x:x+width]
    return img, mask

# transform image into crops of specified size and stride

def forward_crop(img, window, channels, stride):
    crops = np.empty((((img.shape[0] - window[0])//stride + 1 + (img.shape[0] - window[0])%stride)*((img.shape[1] - window[1])//stride + 1 + (img.shape[1] - window[1])%stride), 32, 32, channels))
    idx=0
    stride_i = stride
    stride_j = stride
    ini_i = 0
    ini_j = 0
    for j in range((img.shape[1]- window[1])//stride + 1 +(img.shape[1] - window[1])%stride):
        for i in range((img.shape[0]- window[0])//stride + 1 +(img.shape[0] - window[0])%stride):
            crops[idx] = img[ini_i:window[0]+ini_i, ini_j:window[1]+ini_j].copy()
            if ini_i + window[0] + stride > img.shape[0]:
                stride_i = 1
            else:
                stride_i = stride
            if ini_j + window[1] + stride > img.shape[1]:
                stride_j = 1
            else:
                stride_j = stride
            ini_i += stride_i
            idx += 1
        ini_i = 0
        ini_j += stride_j
    return crops

# reconstruct the crops into the original image. 
# any intersection between the crops (determined by the stride) is the mean between the pixels in it. 

def reconstruct(crops, img_size, window, channels, stride):
    img = np.zeros((img_size[0], img_size[1], channels))
    sum_it = np.zeros((img_size[0], img_size[1], channels))
    idx=0
    stride_i = stride
    stride_j = stride
    ini_i = 0
    ini_j = 0
    for j in range((img.shape[1]- window[1])//stride + 1 + (img.shape[1] - window[1])%stride):
        for i in range((img.shape[0]- window[0])//stride + 1 + (img.shape[0] - window[0])%stride):
            img[ini_i:window[0]+ini_i, ini_j:window[1]+ini_j] += crops[idx]
            sum_it[ini_i:window[0]+ini_i, ini_j:window[1]+ini_j] += 1
            if ini_i + window[0] + stride > img.shape[0]:
                stride_i = 1
            else:
                stride_i = stride
            if ini_j + window[1] + stride > img.shape[1]:
                stride_j = 1
            else:
                stride_j = stride
            ini_i += stride_i
            idx += 1 
        ini_i = 0
        ini_j += stride_j  
    return img*(1/sum_it)
