''''
    From:
        https://github.com/MahatmaSun1/SaConvSLTM/blob/041ecb020d151a21b4a1c3426c2e4e56269c5bff/SaConvSLTM/SaConvLSTM.py
    Paper:
        https://ojs.aaai.org//index.php/AAAI/article/view/6819
'''
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import tensorflow.compat.v2 as tf

import numpy as np

import tensorflow.keras as keras
from keras import backend as K
from keras.utils import conv_utils
from keras.layers.rnn.dropout_rnn_cell_mixin import DropoutRNNCellMixin
from keras.layers.rnn.base_conv_rnn import ConvRNN
from tensorflow.keras.layers import Layer
#from tensorflow.keras.engine.base_layer import base_layer

tf.executing_eagerly()

class SaConvLSTM2DCell(DropoutRNNCellMixin, Layer): #, base_layer.BaseRandomLayer
    """Cell class for the SaConvLSTM2D layer. Modified from `tensorflow.python.keras.layers.convolutional_recurrent.ConvLSTM2DCell`
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 rank=2,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        
        super(SaConvLSTM2DCell, self).__init__(**kwargs)
        self.filters = filters
        self.rank = rank
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank,
                                                        'dilation_rate')
        self.activation = tf.keras.activations.get(activation)
        self.recurrent_activation = tf.keras.activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.recurrent_initializer = tf.keras.initializers.get(recurrent_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = tf.keras.regularizers.get(recurrent_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.recurrent_constraint = tf.keras.constraints.get(recurrent_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.filters, self.filters, self.filters)

    def build(self, input_shape):
        super().build(input_shape)
        self._shape = input_shape
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters * 4)
        self.kernel_shape = kernel_shape
        recurrent_kernel_shape = self.kernel_size + (self.filters, self.filters * 4)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=recurrent_kernel_shape,
            initializer=self.recurrent_initializer,
            name='recurrent_kernel',
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        # sa weight
        # kernel for self attention
        sa_kernel_shape = (1, 1) + (self.filters, 1 * 5)
        self.sa_kernel = self.add_weight(shape=sa_kernel_shape,
                                         initializer=self.kernel_initializer,
                                         name='sa_kernel',
                                         regularizer=self.kernel_regularizer,
                                         constraint=self.kernel_constraint
                                         )

        kernel_z_shape = self.kernel_size + (2, self.filters)  
        self.kernel_z = self.add_weight(shape=kernel_z_shape,
                                        initializer=self.kernel_initializer,
                                        name='kernel_z_shape',
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)

        depth_wise_kernel_size = self.kernel_size + (self.filters, 6)
        self.depth_wise_kernel = self.add_weight(shape=depth_wise_kernel_size,
                                                 initializer=self.kernel_initializer,
                                                 name='depth_wise_kernel',
                                                 regularizer=self.kernel_regularizer,
                                                 constraint=self.kernel_constraint)

        if self.use_bias:
            self.sa_bias = self.add_weight(
                shape=(self.filters * 3),
                name='bias',
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.sa_bias = None
        # sa weight end

        if self.use_bias:
            if self.unit_forget_bias:

                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.filters,), *args, **kwargs),
                        tf.keras.initializers.get('ones')((self.filters,), *args, **kwargs),
                        self.bias_initializer((self.filters * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(
                shape=(self.filters * 4,),
                name='bias',
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=None):
        h_tm1 = states[0] 
        c_tm1 = states[1] 

        # dropout matrices for input units
        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        # dropout matrices for recurrent units
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=4)

        if 0 < self.dropout < 1.:
            inputs_i = inputs * dp_mask[0]
            inputs_f = inputs * dp_mask[1]
            inputs_c = inputs * dp_mask[2]
            inputs_o = inputs * dp_mask[3]
        else:
            inputs_i = inputs
            inputs_f = inputs
            inputs_c = inputs
            inputs_o = inputs

        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1

        (kernel_i, kernel_f,
         kernel_c, kernel_o) = tf.split(self.kernel, 4, axis=3)  # (3, 3, input_dim, filters)
        (recurrent_kernel_i,
         recurrent_kernel_f,
         recurrent_kernel_c,
         recurrent_kernel_o) = tf.split(self.recurrent_kernel, 4, axis=3)

        if self.use_bias:
            bias_i, bias_f, bias_c, bias_o = tf.split(self.bias, 4)
        else:
            bias_i, bias_f, bias_c, bias_o = None, None, None, None

        # input_i: batch
        x_i = self.input_conv(inputs_i, kernel_i, bias_i, padding=self.padding)
        x_f = self.input_conv(inputs_f, kernel_f, bias_f, padding=self.padding)
        x_c = self.input_conv(inputs_c, kernel_c, bias_c, padding=self.padding)
        x_o = self.input_conv(inputs_o, kernel_o, bias_o, padding=self.padding)
        h_i = self.recurrent_conv(h_tm1_i, recurrent_kernel_i)
        h_f = self.recurrent_conv(h_tm1_f, recurrent_kernel_f)
        h_c = self.recurrent_conv(h_tm1_c, recurrent_kernel_c)
        h_o = self.recurrent_conv(h_tm1_o, recurrent_kernel_o)

        i = self.recurrent_activation(x_i + h_i)
        f = self.recurrent_activation(x_f + h_f)
        c = f * c_tm1 + i * self.activation(x_c + h_c)
        o = self.recurrent_activation(x_o + h_o)
        h = o * self.activation(c)

        # sa computation
        m_t_minus_one = states[2]  # h, w, filters
        h_t, c_t = h, c

        (kernel_hv, kernel_hk,
         kernel_hq, kernel_mk,
         kernel_mv) = tf.split(self.sa_kernel, 5, axis=3)  # kernel_size, filters, 1, turn to one layer

        if self.use_bias:
            bias_i, bias_g, bias_o = tf.split(self.sa_bias, 3)
        else:
            bias_i, bias_g, bias_o = None, None, None

        v_h = self.sa_conv(h_t, kernel_hv)
        k_h = self.sa_conv(h_t, kernel_hk)
        q_h = self.sa_conv(h_t, kernel_hq)
        k_m = self.sa_conv(m_t_minus_one, kernel_mk)
        v_m = self.sa_conv(m_t_minus_one, kernel_mv)  # h, w, 1

        q_h = K.squeeze(q_h, 3)
        k_m = K.squeeze(k_m, 3)
        k_h = K.squeeze(k_h, 3)

        e_m = tf.matmul(q_h, k_m)
        alpha_m = K.softmax(e_m)
        e_h = tf.matmul(q_h, k_h)
        alpha_h = K.softmax(e_h)


        v_m = K.squeeze(v_m, 3)
        v_h = K.squeeze(v_h, 3)
        z_m = tf.matmul(alpha_m, v_m)
        z_h = tf.matmul(alpha_h, v_h)

        z_m = K.expand_dims(z_m, 3)
        z_h = K.expand_dims(z_h, 3)
        zi = self.sa_conv(K.concatenate((z_h, z_m), 3), self.kernel_z)

        (kernel_m_zi, kernel_m_hi,
         kernel_m_zg, kernel_m_hg,
         kernel_m_zo, kernel_m_ho) = tf.split(self.depth_wise_kernel, 6, axis=3)  #

        i = K.sigmoid(K.depthwise_conv2d(zi, kernel_m_zi, padding='same') + K.depthwise_conv2d(h_t, kernel_m_hi, padding='same') + bias_i)
        g = K.tanh(K.depthwise_conv2d(zi, kernel_m_zg, padding='same') + K.depthwise_conv2d(h_t, kernel_m_hg, padding='same') + bias_g)
        o = K.sigmoid(K.depthwise_conv2d(zi, kernel_m_zo, padding='same') + K.depthwise_conv2d(h_t, kernel_m_ho, padding='same') + bias_o)

        m_t = (1 - i) * m_t_minus_one + i * g
        h_hat_t = m_t * o
        # sa computation end
        return h_hat_t, [c_t, h_hat_t, m_t]
    
    @property
    def _conv_func(self):
        if self.rank == 1:
            return K.conv1d
        if self.rank == 2:
            return K.conv2d
        if self.rank == 3:
            return K.conv3d

    def input_conv(self, x, w, b=None, padding='valid'):
        conv_out = self._conv_func(x, w, strides=self.strides,
                            padding=padding,
                            data_format=self.data_format,
                            dilation_rate=self.dilation_rate)
        if b is not None:
            conv_out = K.bias_add(conv_out, b,
                                data_format=self.data_format)
        return conv_out

    def recurrent_conv(self, x, w):
        conv_out = self._conv_func(x, w, strides=(1, 1),
                            padding='same',
                            data_format=self.data_format)
        return conv_out

    # my conv
    def sa_conv(self, x, w):
        conv_out = self._conv_func(x, w, strides=(1, 1),
                            padding='same',
                            data_format=self.data_format)
        return conv_out

    # my conv end

    def get_config(self):
        config = {'rank': self.rank,
                  'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': tf.keras.activations.serialize(self.activation),
                  'recurrent_activation': tf.keras.activations.serialize(
                      self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': tf.keras.initializers.serialize(
                      self.kernel_initializer),
                  'recurrent_initializer': tf.keras.initializers.serialize(
                      self.recurrent_initializer),
                  'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': tf.keras.regularizers.serialize(
                      self.kernel_regularizer),
                  'recurrent_regularizer': tf.keras.regularizers.serialize(
                      self.recurrent_regularizer),
                  'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': tf.keras.constraints.serialize(
                      self.kernel_constraint),
                  'recurrent_constraint': tf.keras.constraints.serialize(
                      self.recurrent_constraint),
                  'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        
        base_config = super(SaConvLSTM2DCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SaConvLSTM2D(ConvRNN):
    def __init__(self,
                 filters,
                 kernel_size,
                 rank=2,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        cell = SaConvLSTM2DCell(filters=filters,
                                rank=rank,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                data_format=data_format,
                                dilation_rate=dilation_rate,
                                activation=activation,
                                recurrent_activation=recurrent_activation,
                                use_bias=use_bias,
                                kernel_initializer=kernel_initializer,
                                recurrent_initializer=recurrent_initializer,
                                bias_initializer=bias_initializer,
                                unit_forget_bias=unit_forget_bias,
                                kernel_regularizer=kernel_regularizer,
                                recurrent_regularizer=recurrent_regularizer,
                                bias_regularizer=bias_regularizer,
                                kernel_constraint=kernel_constraint,
                                recurrent_constraint=recurrent_constraint,
                                bias_constraint=bias_constraint,
                                dropout=dropout,
                                recurrent_dropout=recurrent_dropout,
                                dtype=kwargs.get('dtype'))
        super().__init__(rank=rank,
                         cell=cell,
                         return_sequences=return_sequences,
                         return_state=return_state,
                         go_backwards=go_backwards,
                         stateful=stateful,**kwargs)
        
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # self._maybe_reset_cell_dropout_mask(self.cell)
        return super(SaConvLSTM2D, self).call(inputs,
                                              mask=mask,
                                              training=training,
                                              initial_state=initial_state)

    @property
    def filters(self):
        return self.cell.filters

    @property
    def kernel_size(self):
        return self.cell.kernel_size

    @property
    def strides(self):
        return self.cell.strides

    @property
    def padding(self):
        return self.cell.padding

    @property
    def data_format(self):
        return self.cell.data_format

    @property
    def dilation_rate(self):
        return self.cell.dilation_rate

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {'rank': self.rank,
                  'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': tf.keras.activations.serialize(self.activation),
                  'recurrent_activation': tf.keras.activations.serialize(
                      self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': tf.keras.initializers.serialize(
                      self.kernel_initializer),
                  'recurrent_initializer': tf.keras.initializers.serialize(
                      self.recurrent_initializer),
                  'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': tf.keras.regularizers.serialize(
                      self.kernel_regularizer),
                  'recurrent_regularizer': tf.keras.regularizers.serialize(
                      self.recurrent_regularizer),
                  'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': tf.keras.regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': tf.keras.constraints.serialize(
                      self.kernel_constraint),
                  'recurrent_constraint': tf.keras.constraints.serialize(
                      self.recurrent_constraint),
                  'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(SaConvLSTM2D, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)