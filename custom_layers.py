#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:07:31 2019

@author: berend
"""

from keras import backend as K
from keras import layers
from keras.layers import Layer
import tensorflow as tf

class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [layers.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        o = [s[0]]
        if s[1] is None:
            o.append(s[1])
        else:
            o.append(s[1] + 2 * self.padding[0])
        if s[2] is None:
            o.append(s[2])
        else:
            o.append(s[2] + 2 * self.padding[1])
        o.append(s[3])
        
        return tuple(o)

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')
    
class InstanceNormalization(Layer):

    def __init__(self, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.gamma = self.add_weight('gamma', shape=(input_shape[3],), initializer="ones", trainable=True)
        self.beta = self.add_weight('beta', shape=(input_shape[3],), initializer="zero", trainable=True)
        self.epsilon = 1e-3
        super(InstanceNormalization, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        
        input_shape = K.int_shape(inputs)
        
        reduction_axes = [1,2]
        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev
        
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[3] = input_shape[3]

        if self.gamma:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.beta:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed        
        
        
    def compute_output_shape(self, input_shape):
        return input_shape    
    
    
    
def Conv2DReflect(y, nb_channels, kernel_size = (3,3), strides=(1,1), kernel_initializer ='glorot_uniform'):
    
    padding = tuple([(i-1)//2 for i in kernel_size])
    
    y = ReflectionPadding2D(padding)(y)
    
    y = layers.Conv2D(nb_channels, kernel_size = kernel_size, strides=strides, padding = 'valid', kernel_initializer = kernel_initializer)(y)
    
    return y


def residual_block(y, nb_channels):
    shortcut = y

    y = Conv2DReflect(y, nb_channels, kernel_size=(3, 3),strides=(1, 1))
    y = InstanceNormalization()(y)
    y = layers.ReLU()(y)


    y = Conv2DReflect(y, nb_channels, kernel_size=(3, 3),strides=(1, 1))
    #y = layers.BatchNormalization()(y)
    #no relu on the second layer
    
    y = layers.add([shortcut, y])
    
    return y