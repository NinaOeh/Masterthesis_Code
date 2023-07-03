# DINCAEv1: Data-Interpolating Convolutional Auto-Encoder
# Copyright (C) 2019 Alexander Barth

# DINCAEv2: Data-Interpolating Convolutional Auto-Encoder
# Copyright (C) 2022 Alexander Barth

"""
DINCAE (Data-Interpolating Convolutional Auto-Encoder) is a neural network to
reconstruct missing data in satellite observations.

The code for version 1 is available at:
[https://github.com/gher-ulg/DINCAE](https://github.com/gher-ulg/DINCAE)

The code for version 2 in Julia is available at:
[https://github.com/gher-uliege/DINCAE.jl](https://github.com/gher-uliege/DINCAE.jl)

This is version 2 of DINCAE in Python. It is based on TensorFlow 2.0 and   
uses the Dataset API for data loading and training.
"""
#%%

import os
import random
from math import ceil, floor
from netCDF4 import Dataset, num2date
import numpy as np
import tensorflow as tf
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import json

logger = logging.getLogger('root')
FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


def identity(x):
    return x

# save inversion
def sinv(x, minx = 1e-3):
    return 1 / tf.maximum(x,minx)


class DINCAE(object):
    def __init__(self, nx, ny, ntime, resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                 nvar = 10, enc_nfilter_internal = [16, 30, 58, 110, 209], learning_rate = 1e-3,
                 iseed = None, nepoch_keep_missing = 0, net_name = "DINCAE", alpha = 0.5):
        """
        Train a neural network to reconstruct missing data. The network is a
        convolutional auto-encoder with skip connections. 

        Parameters:
        -----------
        nx : int
            Number of pixels in x-direction
        ny : int
            Number of pixels in y-direction
        ntime : int
            Number of time steps
        resize_method : tf.image.ResizeMethod
            Method used for resizing the input data 
        nvar : int  
            Number of input variables
        enc_nfilter_internal : list
            Number of filters in the internal layers of the encoder
        learning_rate : float
            Learning rate for the Adam optimizer
        iseed : int
            Seed for the random number generator
        nepoch_keep_missing : int
            Number of epochs to keep the missing data
        net_name : str
            Name of the network
        alpha : float
            Ratio between intermediate and final output of the network
        """


        if iseed != None:
            np.random.seed(iseed)
            tf.compat.v1.set_random_seed(np.random.randint(0,2**32-1))
            random.seed(np.random.randint(0,2**32-1))

        print("enc_nfilter_internal ",enc_nfilter_internal)
        print("nvar ",nvar)
        print("nepoch_keep_missing ",nepoch_keep_missing)
        
        # set the weights for the cost function
        self.alpha1 = alpha
        self.alpha2 = 1-alpha

        # number of output variables
        self.nvarout = 2
        self.enc_nfilter_internal = enc_nfilter_internal
        
        # define filters for both autoencoders - with refinement
        self.enc_nfilter_1 = [nvar] + enc_nfilter_internal
        self.dec_nfilter_1 = enc_nfilter_internal[::-1] + [self.nvarout]
        
        self.enc_nfilter_2 = [self.nvarout] + enc_nfilter_internal
        self.dec_nfilter_2 = enc_nfilter_internal[::-1] + [self.nvarout]
        
        self.net_name = net_name        
        self.nx = nx
        self.ny = ny
        
        self.ntime = ntime
        
        self.learning_rate = learning_rate
        self.resize_method = resize_method
        
        self.model, intermediate_output = self.build_unet()
        self.compile_unet(self.model, intermediate_output)

    def build_unet(self):
        
        #Input
        inputs_ = tf.keras.Input((self.nx, self.ny, self.ntime*2 + 4), name='inputs_img')
        # activation function for convolutional layer
        conv_activation = tf.nn.leaky_relu

        # Encoder Function
        
        def encoder_layer(l, input_, enc_nfilter):
            conv = tf.keras.layers.Conv2D(enc_nfilter,
                                        kernel_size=(3, 3),
                                        padding='same',
                                        activation=conv_activation,
                                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01))(input_)
            print("encoder: output size of convolutional layer: ", l, conv.shape)

            avgpool = tf.keras.layers.AveragePooling2D(pool_size=(2, 2),
                                                    strides=(2, 2),
                                                    padding='same')(conv)
            print("encoder: output size of pooling layer: ", l, avgpool.shape)

            return conv, avgpool
        
        # Decoder Function
        
        def decoder_layer(l, input_, dec_nfilter, enc_avgpool):
            upsample = tf.keras.layers.UpSampling2D(size=(2, 2),
                                                    interpolation=self.resize_method)(input_)
            print("decoder: output size of upsample layer: ", l, upsample.shape)

            conv = tf.keras.layers.Conv2D(dec_nfilter,
                                        kernel_size=(3, 3),
                                        padding='same',
                                        activation=conv_activation,
                                        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01))(upsample)
            print("decoder: output size of convolutional layer: ", l, conv.shape)
            
            added = tf.keras.layers.Add()([conv, enc_avgpool])
            print("decoder: output size of addition: ", l, added.shape)

            return upsample, added
        
        enc_nlayers = len(self.enc_nfilter_1)
        
        ### Run the first Encoder
        
        print("Start of the first Autoencoder")
    
        enc_conv_1 = [None] * enc_nlayers
        enc_avgpool_1 = [None] * enc_nlayers

        enc_avgpool_1[0] = inputs_
        
        for l in range(1, enc_nlayers):
            enc_conv_1[l], enc_avgpool_1[l] = encoder_layer(l, enc_avgpool_1[l-1], self.enc_nfilter_1[l])

        enc_last_1 = enc_avgpool_1[-1]


        ### Run the first Decoder

        dec_conv_1 = [None] * enc_nlayers
        dec_upsample_1 = [None] * enc_nlayers

        dec_conv_1[0] = enc_last_1
        
        
        for l in range(1, enc_nlayers-1):
            l2 = enc_nlayers - l
            dec_upsample_1[l], dec_conv_1[l] = decoder_layer(l, dec_conv_1[l-1], 
                                                             self.dec_nfilter_1[l], 
                                                             enc_avgpool_1[l2-1])

        # last layer of first decoder
        upsample_1 = tf.keras.layers.UpSampling2D(size=(2, 2),
                                                    interpolation=self.resize_method)(dec_conv_1[-2])
        print("decoder: output size of upsample layer: ", upsample_1.shape)
        intermediate_output = tf.keras.layers.Conv2D(self.dec_nfilter_1[enc_nlayers-1],
                                    kernel_size=(3, 3),
                                    padding='same',
                                    activation=conv_activation,
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01))(upsample_1)
        print("decoder: intermediate output size: ", intermediate_output.shape)
        
        # Concatenate intermediate output with the initial input
        second_input = tf.keras.layers.Concatenate()([inputs_, intermediate_output])
        # print("Concatenating the intermediate output and the initial input: size: ", second_input.shape)  
        
        ### Run the second Encoder
        
        print("Start of the second Autoencoder (Refinement)")
    
        enc_conv_2 = [None] * enc_nlayers
        enc_avgpool_2 = [None] * enc_nlayers

        enc_avgpool_2[0] = intermediate_output
        
        for l in range(1, enc_nlayers):
            enc_conv_2[l], enc_avgpool_2[l] = encoder_layer(l, enc_avgpool_2[l-1], self.enc_nfilter_2[l])

        enc_last_2 = enc_avgpool_2[-1]


        ### Run the second Decoder

        dec_conv_2 = [None] * enc_nlayers
        dec_upsample_2 = [None] * enc_nlayers

        dec_conv_2[0] = enc_last_2
        
        
        for l in range(1, enc_nlayers-1):
            l2 = enc_nlayers - l
            dec_upsample_2[l], dec_conv_2[l] = decoder_layer(l, dec_conv_2[l-1], 
                                                             self.dec_nfilter_2[l], 
                                                             enc_avgpool_2[l2-1])

        # last layer of second decoder
        upsample_2 = tf.keras.layers.UpSampling2D(size=(2, 2),
                                                    interpolation=self.resize_method)(dec_conv_2[-2])
        print("decoder: output size of upsample layer: ", upsample_2.shape)
        output = tf.keras.layers.Conv2D(self.dec_nfilter_2[enc_nlayers-1],
                                    kernel_size=(3, 3),
                                    padding='same',
                                    activation=conv_activation,
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.02))(upsample_2)
        print("decoder: final output size: ", intermediate_output.shape)
        
        
        model = tf.keras.Model(inputs=inputs_, outputs=output)
        
        return model, intermediate_output
    
    def compile_unet(self, model, intermediate_output):
        model.compile(
                optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = self.learning_rate),
                loss=self.loss_total(intermediate_output),
                metrics=[self.loss_mae()]
            )
    
    def loss_total(self, intermediate_output):
        """
        Creates a loss function
        """
        def loglike_loss(y_true, y_pred):
            loginvσ2_rec = y_pred[:,:,:,1]
            invσ2_rec = tf.exp(tf.minimum(loginvσ2_rec,10))
            σ2_rec = sinv(invσ2_rec)
            m_rec = y_pred[:,:,:,0]


            σ2_true = sinv(y_true[:,:,:,1])
            m_true = y_true[:,:,:,0] * σ2_true

            #difference = tf.minimum(m_rec - m_true, 100)
            difference = m_rec - m_true
            
            mask_noncloud = tf.cast(tf.math.logical_not(tf.equal(y_true[:,:,:,1], 0)),
                                y_true.dtype)

            n_noncloud = tf.reduce_sum(input_tensor=mask_noncloud)
            
            cost = (tf.reduce_sum(input_tensor=tf.multiply(tf.math.log(σ2_rec),mask_noncloud)) +
                    tf.reduce_sum(input_tensor=tf.multiply(difference**2 / σ2_rec,mask_noncloud))) / tf.maximum(n_noncloud,1)
            
            return cost
        
        
        def loss(y_true, y_pred):
            intermediate_cost = loglike_loss(y_true, intermediate_output)
            finished_cost = loglike_loss(y_true, y_pred)
            cost = self.alpha1*intermediate_cost + self.alpha2*finished_cost
            return cost
        
        loss.__name__ = 'loss_total'
        
        return loss
    
    def loss_mae(self):
        """
        Creates a loss function
        """
        def loss(y_true, y_pred):
            loginvσ2_rec = y_pred[:,:,:,1]
            invσ2_rec = tf.exp(tf.minimum(loginvσ2_rec,10))
            σ2_rec = sinv(invσ2_rec)
            m_rec = y_pred[:,:,:,0] * σ2_rec


            σ2_true = sinv(y_true[:,:,:,1])
            m_true = y_true[:,:,:,0] * σ2_true

            difference = m_rec - m_true
            
            mask_noncloud = tf.cast(tf.math.logical_not(tf.equal(y_true[:,:,:,1], 0)),
                                y_true.dtype)

            n_noncloud = tf.reduce_sum(input_tensor=mask_noncloud)
            
            RMSE = tf.sqrt(tf.reduce_sum(input_tensor=tf.multiply(difference**2,mask_noncloud))
                / tf.maximum(n_noncloud,1))
            
            return RMSE
        loss.__name__ = 'rmse_loss'
        
        return loss
    
    def summary(self):
        """Get summary of the UNet model"""
        print(self.model.summary())


    
    def predict(self, sample, **kwargs):
        """Run prediction using this model"""
        return self.model.predict(sample, **kwargs)
    
    def save(self, name, save_weights = True):
        if save_weights:
            self.model.save_weights(name)
        data = {}
        data['nx'] =  self.nx
        data['ny'] =  self.ny
        data['ntime'] =  self.ntime
        data['resize_method'] =  self.resize_method
        data['learning_rate'] = self.learning_rate
        data['nvarout'] = self.nvarout
        data['enc_nfilter_internal'] = self.enc_nfilter_internal
        data["net_name"] = self.net_name
        
        with open(name + ".json", 'w') as fp:
            json.dump(data, fp)
            
    @staticmethod
    def load(name, weights_name = "", inference_only=True, dilation_rate = 1):
        with open(name + ".json", 'r') as fp:
            data = json.load(fp)
        print("I am loading the following data: ")
        print(name)    
        print(data)
      
        m = DINCAE(nx=data["nx"], ny=data["ny"], ntime=data["ntime"], 
                   net_name=data["net_name"], resize_method=data['resize_method'], 
                   learning_rate = data["learning_rate"], enc_nfilter_internal=data['enc_nfilter_internal'])
        
        if not weights_name:
            load_status = m.model.load_weights(name)     
        else:
            load_status = m.model.load_weights(weights_name)     
              
        return m
