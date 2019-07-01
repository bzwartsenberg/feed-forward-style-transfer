#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:12:38 2019

@author: berend
"""
from keras import backend as K
from keras import Model
import keras
import tensorflow as tf

def get_vgg_loss_model():
    """ Gets VGG16 model and extract layers for the loss functions
    """
    
    vgg = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    # Content layer where will pull our feature maps
    content_layers = ['block5_conv2'] 

    # Style layer we are interested in
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1'
                   ]        
        
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    
    nstyle = len(style_layers)
    ncontent = len(content_layers)

    return Model(vgg.input, model_outputs),nstyle,ncontent


def gram_matrix(input_tensor):
    # We make the image channels first 
#     print(input_tensor.shape)
    channels = int(input_tensor.shape[3])
#     total_size = int(input_tensor.shape[2])*int(input_tensor.shape[1])
    a = K.reshape(input_tensor, [K.shape(input_tensor)[0],-1, channels])
    n = K.cast(K.shape(a)[1], tf.float32) ## note: should this be set to trainable, false? Note: checked trainable, this is not one of them
    gram = tf.matmul(a, a, transpose_a=True) ### note: does this do the right matrix multiplication with a batch? I think it works
    return gram / (n * K.cast(channels, tf.float32))

def get_gram_loss(style_feature, style_gram_target):
    """style_feature is a (batch, imx, imy, chan) tensor,
    style_gram_target is a (1, 64, 64 tensor)"""

    style_gram = gram_matrix(style_feature) #gets a (batch, 64, 64 tensor)

    return tf.reduce_mean(tf.square(style_gram - style_gram_target)) #this should broadcast appropriately

def get_style_loss(style_features, style_grams_target):
    
    weight_per_style_layer = 1.0 / float(len(style_features))
    
    style_loss = 0
    
    for style_feature, style_gram_target in zip(style_features, style_grams_target):
        style_loss += weight_per_style_layer * get_gram_loss(style_feature, style_gram_target)
        
    return style_loss


def get_content_loss(content, target_content):
    """content is a [batch,imx,imy,chan] tensor, content is a [batch,imx,imy,chan] tensor"""
    return tf.reduce_mean(tf.square(content - target_content))




def get_loss_func(style_image, style_weight = 1e-2, content_weight = 1e3):
    """style_image as a 4D tensor (1,256,256,3)"""
    
    
    
    
    model,nstyle,ncontent = get_vgg_loss_model()
    
    for layer in model.layers:
        layer.trainable = False
    
    
    
    style_target_features = model.predict(style_image)
        
    style_grams_target = [gram_matrix(image_feature) for image_feature in style_target_features[:nstyle]]
    
    #note: the style_target_features are giant, may need to delete those
    
    def cl(original, generated):

        
        original_features = model(original)
        generated_features = model(generated)
    
        original_content = original_features[-1]
        generated_content = generated_features[-1]
    
        cl = get_content_loss(generated_content, original_content)

        return cl*content_weight
    
    
    def sl(original, generated):

        
        generated_features = model(generated)
    
    
        generated_style_features = [image_feature for image_feature in generated_features[:nstyle]]
        sl = get_style_loss(generated_style_features, style_grams_target)

        return sl*style_weight
    
        
    
    def lossfun(original, generated):
        """original and generated are both [batch,imx,imy,3] shape tensors"""
                
        
        #generate features
        original_features = model(original)
        generated_features = model(generated)
        
        original_content = original_features[-1]
        generated_content = generated_features[-1]
        
        generated_style_features = [image_feature for image_feature in generated_features[:nstyle]]
        
        cl = get_content_loss(generated_content, original_content)
        
        sl = get_style_loss(generated_style_features, style_grams_target)
                
            
        return  cl*content_weight + style_weight*sl

    return lossfun, cl, sl


