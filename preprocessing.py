#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 13:50:50 2019

@author: berend
"""


#load and preprocessing

import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

from keras.preprocessing import image as kp_image
import keras

def process_img(img):
    img = keras.applications.vgg19.preprocess_input(img)
    return img

def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]")
    if len(x.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")
  
    # perform the inverse of the preprocessiing step
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


def load_img(path_to_img, resize = 256, crop = True):
    
    img = Image.open(path_to_img)

    if resize:
        max_dim = resize
        short = min(img.size)
        scale = max_dim/short
        img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
        ## note: may need to crop instead of squeeze
        #crop:
        if crop:
            x_start = max(0, img.size[0]//2 - 128)
            x_end = min(img.size[0], img.size[0]//2 + 128)
            y_start = max(0, img.size[1]//2 - 128)
            y_end = min(img.size[1], img.size[1]//2 + 128)
            img = img.crop([x_start, y_start, x_end, y_end])
        
    arr = kp_image.img_to_array(img)
    
    if arr.shape[2] == 1:
        arr = kp_image.img_to_array(img.convert('RGB'))

    # We need to broadcast the image array such that it has a batch dimension 
    img = np.expand_dims(arr, axis=0)
    return img


def load_and_process_img(path_to_img, resize = 256, crop = True):
    img = load_img(path_to_img, resize = resize, crop = crop)
    img = keras.applications.vgg19.preprocess_input(img)
    return img


def save_and_deprocess_img(out_im, save_path):
    pil_im = Image.fromarray(deprocess_img(out_im))
    pil_im.save(save_path)


def load_train_data(num_images = 1000, files = None):
    
    data = np.zeros((num_images, 256, 256, 3), dtype = np.float32)
    
    if files is None:
        file_list = os.listdir('train2014/')
    else:
        file_list = files
    
    for i in range(num_images):
        if i % 1000 == 0:
            print('Image {}'.format(i))
        data[i:i+1] = load_and_process_img('train2014/' + file_list[i])
        
        
    return data


def imshow(img, ax = None):
    # Remove the batch dimension
    
    img = deprocess_img(img)
    
    if len(img.shape) == 4:
        out = np.squeeze(img, axis=0)
    else:
        out = img
    # Normalize for display 
    out = out.astype('uint8')
    if ax is None:
        fig,ax = plt.subplots(figsize = (5,5))
    ax.imshow(out)
