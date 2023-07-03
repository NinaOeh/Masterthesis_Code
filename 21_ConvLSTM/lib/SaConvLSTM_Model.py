
import tensorflow.compat.v2 as tf
from lib.SaConvLSTM import SaConvLSTM2D
from tensorflow.python.framework.ops import disable_eager_execution
from numpy.random import seed
seed(2)
tf.keras.utils.set_random_seed(2)
tf.config.experimental.enable_op_determinism()

disable_eager_execution()

import os
os.environ['TF_KERAS'] = '1'


def Sa_build_model(shape, learning_rate, filter=0, 
                  kernel_size_1=3,
                  kernel_size_2=3,
                  kernel_size_3=3,
                  kernel_size_final=3, 
                  act="tanh"):
    
    """
    Function to construct a ConvLSTM model with 3 ConvLSTM layers and 1 Conv3D layer.
    
    Parameters
    ----------
    shape : tuple
        Shape of the input data (x, y, t).
    learning_rate : float
        Learning rate for the Adam optimizer.
    filter : int
        Index of the filter option to use. Default is 0.
    kernel_size_1 : int
        Kernel size for the first ConvLSTM layer. Default is 3.
    kernel_size_2 : int
        Kernel size for the second ConvLSTM layer. Default is 3.
    kernel_size_3 : int
        Kernel size for the third ConvLSTM layer. Default is 3.
    kernel_size_final : int
        Kernel size for the final Conv3D layer. Default is 3.
    act : str
        Activation function to use. Default is "tanh".
    
    """

    
    filter_opts = [(16, 32, 64), (16,16,16), (32, 32, 32), (64, 64, 64)]
    filters = filter_opts[filter]
    
    # Construct the input layer with no definite frame size.
    inp = tf.keras.layers.Input(shape=(None, *shape))

    print(inp.shape)
    

    # Constructing 3 `ConvLSTM2D` layers with batch normalization,
    # followed by a `Conv3D` layer for the spatiotemporal outputs.
    x = SaConvLSTM2D(rank=2, filters=filters[0], 
                           kernel_size=(kernel_size_1, kernel_size_1), 
                           activation=act,
                           padding="same", 
                           return_sequences=True)(inp)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = SaConvLSTM2D(rank=2, filters=filters[1], 
                           kernel_size=(kernel_size_2, kernel_size_2), 
                           activation=act,
                           padding="same", 
                           return_sequences=True)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = SaConvLSTM2D(rank=2, filters=filters[2], 
                           kernel_size=(kernel_size_3, kernel_size_3), 
                           activation=act,
                           padding="same", 
                           return_sequences=True)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    print(f" Shape after last ConvLSTM3d and batch normalisation: {x.shape}")
    
    # output = []
    
    # x_temp = x
    
    # for i in range(5):
    #     x_temp = tf.keras.layers.Conv3D(
    #         filters=1, kernel_size=(kernel_size_final, kernel_size_final, kernel_size_final), 
    #         activation="LeakyReLU", padding="same"
    #     )(x_temp)
    #     output.append(x_temp)

    output = tf.keras.layers.Conv3D(
        filters=5, kernel_size=(kernel_size_final, kernel_size_final, kernel_size_final), 
        activation="linear", padding="same"
    )(x)
    
    #print(f" Shape after last conv 3D : {output.shape}")

    # Next, we will build the complete model and compile it.
    model = tf.keras.models.Model(inp, output)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, clipnorm=1.0)
    print(optimizer.lr)
    model.compile(
        loss=tf.keras.losses.mean_squared_error, 
        optimizer=optimizer, #binary_crossentropy
    )
    
    return model
    
    
