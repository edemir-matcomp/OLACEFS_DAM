from __future__ import division

import os
import random
import skimage
import numpy as np
from tensorflow import keras
from skimage.io import imread, imsave

import utils as U

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_ids, batch_size=1, dim=(32,32), n_channels=2, mean=(0,0) ,std=(1,1), shuffle=True):
        'Initialization'
        self.list_ids = list_ids
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.mean = mean
        self.std = std
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_ids_temp = [self.list_ids[k] for k in indexes]
        x, y = self.__data_generation(list_ids_temp)

        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        'Generates data containing batch_size samples'
        x = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim))

        dataset_mean = self.mean
        dataset_std = self.std

        for i, ID in enumerate(list_ids_temp):

            img1 = skimage.img_as_float64(imread("/home/users/DATASETS/MapBiomas_SAR/GEE_mapbiomas/"+ ID + "_2019-01-01.tif"))  
            img2 = skimage.img_as_float64(imread("/home/users/DATASETS/MapBiomas_SAR/GEE_mapbiomas/"+ ID + "_2019-04-01.tif"))
            img3 = skimage.img_as_float64(imread("/home/users/DATASETS/MapBiomas_SAR/GEE_mapbiomas/"+ ID + "_2019-07-01.tif"))
            img4 = skimage.img_as_float64(imread("/home/users/DATASETS/MapBiomas_SAR/GEE_mapbiomas/"+ ID + "_2019-10-01.tif"))
            img5 = skimage.img_as_float64(imread("/home/users/DATASETS/MapBiomas_SAR/GEE_mapbiomas/"+ ID + "_2020-01-01.tif"))   
            mask = skimage.img_as_float64(imread("/home/users/DATASETS/MapBiomas_SAR/GEE_mapbiomas_masks/"+ ID + ".tif"))
            
            img1 = U.normalization(img1, mean=dataset_mean, std=dataset_std)
            img2 = U.normalization(img2, mean=dataset_mean, std=dataset_std)
            img3 = U.normalization(img3, mean=dataset_mean, std=dataset_std)
            img4 = U.normalization(img4, mean=dataset_mean, std=dataset_std)
            img5 = U.normalization(img5, mean=dataset_mean, std=dataset_std)

            img  = np.concatenate((img1,img2,img3,img4,img5),axis=2)

            # 32x32 random crop
            if self.shuffle == True:
                img, mask = U.random_crop(img, mask, 32, 32)
            else:
                img = img[:32,:32]
                mask = mask[:32,:32]

            x[i,] = img
            y[i,] = mask / 255.

        y = np.expand_dims(y, axis=3)

        return x, y