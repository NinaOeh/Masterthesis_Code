"""
# STpConv3D: Partial 3D convolutions
# Copyright (C) 2022 Marius Appel

STpConv3D is a neural network to
reconstruct missing data in satellite observations.

The original code is available at:
[https://github.com/appelmar/STpconv](https://github.com/appelmar/STpconv)

This file has used part of the code from the original repository and modified it to fit the needs of this project.
"""

import os
import sys
import numpy as np
import json

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from lib.STpconvLayer_chl import STpconv
import lib.Losses_chl as losses


class STpconvUnet(object):

    def __init__(self, n_conv_layers = 4, nx=160, ny=160, nt=30, inference_only=False, net_name='default', 
                 kernel_sizes = None, n_filters = None, learning_rate = 0.0001, n_conv_per_block = 1, strides = None,
                 dilation_rate = 1, act="LeakyReLU", l1=0.01, l2=0.01):
        """
        Neural network for spatio-temporal partial convolutions.
            
        Parameters
        ----------
        n_conv_layers : int, optional
            Number of convolutional layers. The default is 4.
        nx : int, optional
            Number of pixels in x-direction. The default is 160.
        ny : int, optional
            Number of pixels in y-direction. The default is 160.
        nt : int, optional
            Number of time steps. The default is 30.
        inference_only : bool, optional
            If True, the network is only used for inference. The default is False.
        net_name : str, optional
            Name of the network. The default is 'default'.
        kernel_sizes : list, optional
            List of kernel sizes for each convolutional layer. The default is None.
        n_filters : list, optional
            List of number of filters for each convolutional layer. The default is None.
        learning_rate : float, optional
            Learning rate. The default is 0.0001.
        n_conv_per_block : int, optional
            Number of convolutions per block. The default is 1.
        strides : list, optional
            List of strides for each convolutional layer. The default is None.
        dilation_rate : int, optional
            Dilation rate. The default is 1.
        act : str, optional
            Activation function. The default is "LeakyReLU".
        l1 : float, optional
            L1 regularization. The default is 0.01.
        l2 : float, optional
            L2 regularization. The default is 0.01.
        
        """
        
        
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.n_conv_layers = n_conv_layers
        self.inference_only = inference_only
        self.net_name = net_name
        self.learning_rate = learning_rate
        self.n_conv_per_block = n_conv_per_block
        self.dilation_rate = dilation_rate
        self.act = act
        self.l1 = l1
        self.l2 = l2

    
        assert n_conv_layers > 1, "n_conv_layers must be > 1"
        
        if not kernel_sizes is None:
            print(len(kernel_sizes))
            assert len(kernel_sizes) == n_conv_layers, "len(kernel_sizes) must equal n_conv_layers"
            
        if not n_filters is None:
            assert len(n_filters) == n_conv_layers, "len(n_filters) must equal n_conv_layers"
            
        if strides is None:
            # derive strides
            self.strides = []
            for i in range(self.n_conv_layers):
                s = [2,2,2]
                if self.nx / (2 ** (i+1)) < 1:
                    s[0] = 1
                if self.ny / (2 ** (i+1)) < 1:
                    s[1] = 1
                if self.nt / (2 ** (i+1)) < 1:
                    s[2] = 1
                self.strides.append(s)
        else:
            self.strides = strides
            
            assert len(self.strides) == n_conv_layers, "len(strides) must equal n_conv_layers"
            
        # set kernel sizes if not provided    
        if not kernel_sizes is None:
            self.kernel_sizes = kernel_sizes
        else:
            self.kernel_sizes = []
            for i in range(self.n_conv_layers):
                s = (3,3,3)
               # if i == 0:
               #     s = [3,3,3]
                self.kernel_sizes.append(s)
        
        # set n_filters if not provided
        if not n_filters is None:
            self.n_filters = n_filters
        else:
            self.n_filters = []
            for i in range(self.n_conv_layers):
                s = 2 ** (i+5) # start with 32 and go up to 256
                s = min(s, 256)
                self.n_filters.append(s)
            
        
        self.model, inputs_mask, validation_mask, land_mask = self.build_pconv_unet()
        self.compile_pconv_unet(self.model, inputs_mask, validation_mask, land_mask)            
        
   
    def build_pconv_unet(self):      

        # INPUTS
        inputs_img = tf.keras.Input((self.nx, self.ny, self.nt, 2), name='inputs_img')
        inputs_mask = tf.keras.Input((self.nx, self.ny, self.nt, 1), name='inputs_mask')
        input_validation_mask = tf.keras.Input((self.nx, self.ny, self.nt, 1), name='validation_mask')
        input_land_mask = tf.keras.Input((self.nx, self.ny, self.nt, 1), name='land_mask')
        
        # ENCODER
        def encoder_layer(img_in, mask_in, filters, kernel_size, strides, dilation_rate):
            if (self.n_conv_per_block > 1):
                print(img_in.shape)
                conv, mask = STpconv(filters, kernel_size, dilation_rate, strides=strides, 
                                     regularizer = tf.keras.regularizers.l1_l2(l1=self.l1, l2=self.l2),
                                     activation = tf.keras.layers.LeakyReLU(alpha=0.3))([img_in, mask_in])
                print(conv.shape)
                conv = tf.keras.layers.BatchNormalization()(conv)
                if (self.n_conv_per_block > 2):
                    for i in range(self.n_conv_per_block - 2):
                        conv, mask = STpconv(filters, kernel_size, strides=(1,1,1),
                                             activation = tf.keras.layers.LeakyReLU(alpha=0.3),
                                             regularizer = tf.keras.regularizers.l1_l2(l1=self.l1, l2=self.l2))([conv, mask])
                        conv = tf.keras.layers.BatchNormalization()(conv)
                        print(conv.shape)
                conv, mask = STpconv(filters, kernel_size, dilation_rate, strides=(1,1,1), 
                                     activation = tf.keras.layers.LeakyReLU(alpha=0.3),
                                     regularizer = tf.keras.regularizers.l1_l2(l1=self.l1, l2=self.l2))([conv, mask])  
                conv = tf.keras.layers.BatchNormalization()(conv)
                print(conv.shape)       
            else:
                conv, mask = STpconv(filters, kernel_size, dilation_rate, strides=strides, 
                                     activation = tf.keras.layers.LeakyReLU(alpha=0.3),
                                     regularizer = tf.keras.regularizers.l1_l2(l1=self.l1, l2=self.l2))([img_in, mask_in])
                conv = tf.keras.layers.BatchNormalization()(conv)
                print(conv.shape)
           
            encoder_layer.counter += 1
            return conv, mask
        encoder_layer.counter = 0
        
        init_img = tf.keras.layers.Conv3D(1,3,activation='relu', padding='same')(inputs_img)
        init_img = tf.keras.layers.BatchNormalization()(init_img)
        prev_img = init_img
        prev_mask = inputs_mask
        e_conv = []
        e_mask = []
        for i in range(self.n_conv_layers):
            prev_img, prev_mask = encoder_layer(prev_img, prev_mask, 
                                                self.n_filters[i], 
                                                self.kernel_sizes[i], 
                                                self.strides[i],
                                                self.dilation_rate)
            e_conv.append(prev_img)
            e_mask.append(prev_mask)
   
        # DECODER
        def decoder_layer(img_in, mask_in, e_conv, e_mask, filters, kernel_size, size, dilation_rate):
            up_img = tf.keras.layers.UpSampling3D(size=size)(img_in)
            up_mask = tf.keras.layers.UpSampling3D(size=size)(mask_in)
            print(e_conv.shape, up_img.shape)
            print(e_mask.shape, up_mask.shape)
            concat_img = tf.keras.layers.Concatenate(axis=4)([e_conv,up_img])
            concat_mask = tf.keras.layers.Concatenate(axis=4)([e_mask,up_mask])
            
            if (self.n_conv_per_block > 1):
                conv, mask = STpconv(filters, kernel_size, dilation_rate,
                                     activation = tf.keras.layers.LeakyReLU(alpha=0.3),
                                     regularizer = tf.keras.regularizers.l1_l2(l1=self.l1, l2=self.l2))([concat_img, concat_mask])
                conv = tf.keras.layers.BatchNormalization()(conv)
                for i in range(self.n_conv_per_block - 1):
                    conv, mask = STpconv(filters, kernel_size, dilation_rate, 
                                         activation = tf.keras.layers.LeakyReLU(alpha=0.3),
                                         regularizer = tf.keras.regularizers.l1_l2(l1=self.l1, l2=self.l2))([conv, mask])   
                    conv = tf.keras.layers.BatchNormalization()(conv)
            else:
                conv, mask = STpconv(filters, kernel_size, dilation_rate, 
                                     activation = tf.keras.layers.LeakyReLU(alpha=0.3),
                                     regularizer = tf.keras.regularizers.l1_l2(l1=self.l1, l2=self.l2))([concat_img, concat_mask])
                conv = tf.keras.layers.BatchNormalization()(conv)
                
            return conv, mask
            
            
        d_conv = []
        d_mask = []
        prev_img = e_conv[self.n_conv_layers - 1]
        prev_mask =  e_mask[self.n_conv_layers - 1]
        for i in reversed(range(self.n_conv_layers)):       
            if i == 0:
                prev_img, prev_mask = decoder_layer(prev_img, prev_mask, init_img, 
                                                    inputs_mask, 1, (3,3,3), 
                                                    self.strides[i], self.dilation_rate) # decoder layers use 3x3x3 kernels only
            else:
                prev_img, prev_mask = decoder_layer(prev_img, prev_mask, e_conv[i-1], 
                                                     e_mask[i-1], self.n_filters[i-1], (3,3,3), 
                                                     self.strides[i], self.dilation_rate) # decoder layers use 3x3x3 kernels only
             
            d_conv.append(prev_img)
            d_mask.append(prev_mask)
             
        drop = tf.keras.layers.Dropout(0.2)(d_conv[len(d_conv)-1])   
        outputs = tf.keras.layers.Conv3D(1, 1, activation = "linear")(drop) 
        
        # Setup the model inputs / outputs
        model = tf.keras.Model(inputs=[inputs_img, inputs_mask, input_validation_mask, input_land_mask], 
                               outputs=outputs)

        return model, inputs_mask, input_validation_mask, input_land_mask   

    def compile_pconv_unet(self, model, inputs_mask, validation_mask, land_mask):
        model.compile(
                optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = self.learning_rate),
                loss=self.loss_total(inputs_mask, validation_mask, land_mask),
                metrics=[losses.mse_gaps(inputs_mask, validation_mask, land_mask), losses.rmse_gaps(inputs_mask, validation_mask, land_mask),  losses.r2_gaps(inputs_mask, validation_mask, land_mask)]
            )

    def loss_total(self, mask, validation_mask, land_mask):
        """
        Creates a loss function
        """
        def loss(y_true, y_pred):
            return losses.mae_gaps(mask, validation_mask, land_mask)(y_true, y_pred) 
        loss.__name__ = 'loss_total'
        
        return loss
        
    def save(self, name, save_weights = True):
        if save_weights:
            self.model.save_weights(name)
        data = {}
        data['nx'] =  self.nx
        data['ny'] =  self.ny
        data['nt'] =  self.nt
        data['n_conv_layers'] =  self.n_conv_layers
        data['inference_only'] =  self.inference_only
        data['net_name'] =  self.net_name
        data['kernel_sizes'] = self.kernel_sizes
        data['n_filters'] = self.n_filters
        data['learning_rate'] = self.learning_rate
        data['n_conv_per_block'] = self.n_conv_per_block
        data['strides'] = self.strides
        
        
        with open(name + ".json", 'w') as fp:
            json.dump(data, fp)
        
        
    @staticmethod
    def load(name, weights_name = "", inference_only=True, dilation_rate = 1):
        with open(name + ".json", 'r') as fp:
            data = json.load(fp)
        if not 'n_conv_per_block' in data: 
            n_conv_per_block = 1
        else:
            n_conv_per_block =  data['n_conv_per_block']
            
        if not 'strides' in data: 
            strides = None
        else:
            strides =  [tuple(x) for x in data["strides"]]     
        
        if not 'kernel_sizes' in data: 
            kernel_sizes = None
        else:
            kernel_sizes =  [tuple(x) for x in data["kernel_sizes"]]  
            
      
        m = STpconvUnet(n_conv_layers = data["n_conv_layers"], nx=data["nx"], ny=data["ny"], 
                        nt=data["nt"], inference_only=inference_only, net_name=data["net_name"],
                        kernel_sizes = kernel_sizes, n_filters = data["n_filters"], 
                        learning_rate = data["learning_rate"],n_conv_per_block = n_conv_per_block, 
                        strides = strides, dilation_rate= dilation_rate)
        
        if not weights_name:
            load_status = m.model.load_weights(name)     
        else:
            load_status = m.model.load_weights(weights_name)     
              
        return m;
        
    def summary(self):
        """Get summary of the UNet model"""
        print(self.model.summary())


    
    def predict(self, sample, **kwargs):
        """Run prediction using this model"""
        return self.model.predict(sample, **kwargs)
