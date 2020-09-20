from __future__ import division

import os
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard,ReduceLROnPlateau
from keras import callbacks
import pickle

import utils as U
import models as M
import dataloader as D

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())
    
# Build model
model = M.BCDU_net_D3(input_size = (32,32,10))
model.summary()

print('Training')

nb_epoch = 400

dataset_add = '/home/users/pedrow/datasets/Mapbiomas_SAR/'

params_tr = {'batch_size': 8,
          'dim': (32,32),
          'n_channels': 10,
          'mean': (-15.34541179, -8.87553847),
          'std': (1.53520544, 1.45585154),
          'shuffle': True}

params_va = {'batch_size': 8,
          'dim': (32,32),
          'n_channels': 10,
          'mean': (-15.34541179, -8.87553847),
          'std': (1.53520544, 1.45585154),
          'shuffle': False}

file = open("train.txt", "r")
tr_list = [line.rstrip('\n') for line in file]
file.close()

file = open("val.txt", "r")
va_list = [line.rstrip('\n') for line in file]
file.close()

training_generator = D.DataGenerator(tr_list, **params_tr)
validation_generator = D.DataGenerator(va_list, **params_va)

mcp_save = ModelCheckpoint('weightsteste', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs=nb_epoch,
                    shuffle=True,
                    verbose=1,
                    callbacks=[mcp_save])
  
print('Trained model saved')
with open('histteste', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
