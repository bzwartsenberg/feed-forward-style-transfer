#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:22:09 2019

@author: berend
"""

import os
import numpy as np
import json

from keras import layers
from keras import Model
from keras import optimizers
from keras.models import load_model


from losses import get_loss_func
from custom_layers import InstanceNormalization, Conv2DReflect,residual_block,ReflectionPadding2D
from preprocessing import load_train_data,save_and_deprocess_img,load_and_process_img

class StyleTransfer():
    
    def __init__(self, style_image_path, path, style_weight=1., content_weight=1., 
                 input_size=(256,256), optimizer=None, sub_path=None,
                 test_photo_path=None, load_model=None, load_only=False):
        """Style transfer class
        Args:
            style_image_path: path to style image
            path: base path
            style_weight: weight for the style loss
            content_weight: weight for the content loss
            input_size: size used for training
            optimizer: pass a keras optimizer, Adam used standard
            sub_path: sub path used for every model, automatically generated if not given
            test_photo_path: a path to a folder of test photos to run on after training
            load_model: path to a pretrained model
            load_only: do not set up with a new directory for training, only prediction"""
            

        if load_model is None:
            self.build_generator_net(input_size=input_size)
        else:
            self.build_from_path(load_model)

        if not load_only:
            self.style_image = load_and_process_img(style_image_path, resize=False)
            
            self.style_image_path = style_image_path
            self.style_weight = style_weight
            self.content_weight = content_weight
            
            self.loss, self.cl, self.sl = get_loss_func(self.style_image, style_weight=style_weight, content_weight=content_weight)        
            
            if optimizer is None:
                self.optimizer = optimizers.Adam
            else:
                self.optimizer = optimizer
                
            self.path = path
            
            if sub_path is None:
                i = 0
                while os.path.exists(self.path + '{:05d}/'.format(i)):
                    i += 1
                self.sub_path = '{:05d}/'.format(i)
                os.mkdir(self.path + self.sub_path)
            self.test_photo_path = test_photo_path
        
    
    
    def build_generator_net(self, input_size=(256,256)):
        
        inp = layers.Input(shape = (input_size[0],input_size[1],3))
      
        #encoder:
        y = Conv2DReflect(inp, 32, kernel_size = (9,9), strides=(1,1))
        y = InstanceNormalization()(y)
        y = layers.ReLU()(y)
    
        y = Conv2DReflect(y, 64, kernel_size = (3,3), strides=(2,2))
        y = InstanceNormalization()(y)
        y = layers.ReLU()(y)
    
        y = Conv2DReflect(y, 128, kernel_size = (3,3), strides=(2,2))
        y = InstanceNormalization()(y)
        y = layers.ReLU()(y)
        
        y = residual_block(y, 128)
        y = residual_block(y, 128)
        y = residual_block(y, 128)
        y = residual_block(y, 128)
        y = residual_block(y, 128)
        
        y = layers.UpSampling2D(size=(2, 2))(y)
        y = Conv2DReflect(y, 64, kernel_size = (3,3), strides=(1,1))
        y = InstanceNormalization()(y)
        y = layers.ReLU()(y)
    
        y = layers.UpSampling2D(size=(2, 2))(y)
        y = Conv2DReflect(y, 32, kernel_size = (3,3), strides=(1,1))
        y = InstanceNormalization()(y)
        y = layers.ReLU()(y)
    
        y = Conv2DReflect(y, 3, kernel_size = (9,9), strides=(1,1))
    #     y = layers.BatchNormalization()(y)
        y = layers.Activation('tanh')(y)
        
    
        
        
        y = layers.Lambda(lambda x: (x * 150))(y) #scale output to imagenet means
        
        self.generator = Model(inp,y)
        
        
    def train(self, files, verbose=1, epochs=2, batch_size=8, decay_lr=[1e-4,1e-5],
                save_checkpoint=20,chunk_size=1000,save_img=True, save_best_loss=True):
        """Train the classifier
        Args:
            files: a list of filepaths
            verbose: verbosity for output generation
            epochs: number of full iterations of the dataset
            batch_size: size of each training barch
            decay_lr: list of learn_rates size of epochs
            save_checkpoint: save the model every n times chunk_size
            chunk_size: number of files to load at once
            save_img: if true, save a training image every chunk
            save_best_loss: save on the best loss"""
        
        iterations = len(files)//chunk_size
        
        self.cl_history = []
        self.sl_history = []
        self.loss_history = []
        self.epochs = epochs
        self.batch_size = batch_size
        self.decay_lr = decay_lr
        self.chunk_size = chunk_size
        
        for j in range(epochs):
            self.generator.compile(loss = self.loss, optimizer = self.optimizer(decay_lr[j]), metrics = [self.cl, self.sl])
    
            for i in range(iterations):
                if verbose:
                    print('Iteration {} out of {}'.format(i, iterations))
                
                data = load_train_data(chunk_size, files = files[i*chunk_size:(i+1)*chunk_size])
                history = self.generator.fit(data, data, batch_size=batch_size, epochs=1, verbose=verbose)
                
                
                self.cl_history.append(history.history['cl'][0])
                self.sl_history.append(history.history['sl'][0])
                self.loss_history.append(history.history['loss'][0])
                
                self.best_loss = np.inf
                self.best_cl = 0
                self.best_sl
                self.best_loss_epoch = 0
                self.best_loss_iter = 0
                    
                if (i % 5) == 0 and save_img:
                    im_save_dir = self.path + self.sub_path + 'im_checkpoints/'
                    if not os.path.isdir(im_save_dir):
                        os.mkdir(im_save_dir)
                
                    pred = self.generator.predict(data[0:1])
                    save_and_deprocess_img(pred[0], 'pred_epoch_{}_iteration_{}.png'.format(j,i))
                    save_and_deprocess_img(data[0], 'data_epoch_{}_iteration_{}.png'.format(j,i))
                    
                
                if (i%save_checkpoint) == 0:
                    self.generator.save(self.path + self.sub_path + 'checkpoint.h5', include_optimizer=False)        
                    
                if save_best_loss and self.loss_history[-1] < self.best_loss:
                    self.best_loss = self.loss_history[-1]
                    self.best_cl = self.cl_history[-1]
                    self.best_sl = self.sl_history[-1]
                    self.best_loss_epoch = j
                    self.best_loss_iter = i
                    self.generator.save(self.path + self.sub_path + 'best_checkpoint.h5', include_optimizer=False)        
                    
                
    def build_from_path(self, path):
        
        self.generator = load_model(path, custom_objects={'ReflectionPadding2D': ReflectionPadding2D,
                                                          'InstanceNormalization': InstanceNormalization})


    def write_json(self):
        """Write a dictionary to be able to log training"""
        
        out_dict = {
                'style_image_path' : self.style_image_path,
                'style_weight' : self.style_weight,
                'content_weight' : self.content_weight,
                'optimizer' : self.optimizer.__name__,


                'epochs' : self.epochs,
                'batch_size' : self.batch_size,
                'chunk_size' : self.chunk_size,
                'decay_lr' : self.decay_lr,
                'save_checkpoint' : self.save_checkpoint,
                
                'cl_history' : self.cl_history,
                'sl_history' : self.sl_history,
                'loss_history' : self.loss_history,
                
                'best_loss' : self.best_loss,
                'best_cl' : self.best_cl,
                'best_sl' : self.best_sl,
                'best_loss_epoch' : self.best_loss_epoch,
                'best_loss_iter' : self.best_loss_iter,
                
                
                }
        
        with open(self.path + self.sub_path + 'log.json','w') as f:
            json.dump(out_dict, f)
        
        #write a dictionary,
        
        #epochs,
        #batch_size
        
        
    
                
        