import os
import shutil
import numpy as np
import tensorflow as tf
import time

# Import modules from lib directory
from lib.STpconvUnet_chl import STpconvUnet
from lib.DataGenerator_chl_nc import DataGenerator

print("Using TensorFlow version", tf.__version__)
import matplotlib.pyplot as plt

from tensorflow.python.framework.ops import disable_eager_execution


print("Starting the training")
disable_eager_execution()
print("...")
# Hyperparameters and other options
BATCH_SIZE = 4
N_EPOCHS = 120 #300
OUT_PATH = os.getcwd()
DATA_PATH_TRAINING = "../CHL_DATA/InputData_GTL/TrainData_NC_GTL"
DATA_PATH_VALIDATION = "../CHL_DATA/InputData_GTL/ValData_NC_GTL"

model_name = "model_213_GTL_dataperc30-100_4"

N_CONV_LAYERS = 4
N_CONV_PER_BLOCK = 1
KERNEL_SIZES = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
DILATION_RATE = 1
N_FILTERS    = [16, 32, 64, 128] #, 128
STRIDES = [(2, 2, 2),(2, 2, 1),(2, 2, 1), (2, 2, 2)] #, (2, 2, 2)


trn_generator = DataGenerator(DATA_PATH_TRAINING, batch_size = BATCH_SIZE)
val_generator = DataGenerator(DATA_PATH_VALIDATION, batch_size = BATCH_SIZE)

model = STpconvUnet(n_conv_layers = N_CONV_LAYERS, nx=160, ny=160, nt=20, 
                    kernel_sizes = KERNEL_SIZES, n_filters = N_FILTERS, 
                    learning_rate=0.005, n_conv_per_block=N_CONV_PER_BLOCK, 
                    dilation_rate=DILATION_RATE, strides = STRIDES,
                    act="LeakyReLU") #, loss="loglike"
model.summary()

# save model at the end of each epoch
checkpoint_dir = os.path.join(OUT_PATH, model_name) 
# Model name convention:

# first letter: 0 - data as-is, 1 - normalized (min-max), 2 - log-transformed, 3 - boxcox transformed
# second letter: 0 - batch size 2, 1 - batch size 4, 2 - batch size 8, 3 - batch size 10
# third lettter: order of strides 0 - 221,221,222,222 ; 1 - 222,222,221,221 ; 2 - 221,222,222,221 , 3 - 222,221,221,222

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=40)

# Reduce learning rate when a metric has stopped improving.
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", patience=10)

if os.path.exists(checkpoint_dir):
    shutil.rmtree(checkpoint_dir)
os.mkdir(checkpoint_dir)
checkpoint_filepath = os.path.join(checkpoint_dir, "epoch_{epoch:02d}.h5")
model_checkpoint_callback = [tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filepath,
    save_weights_only=True,
    save_best_only=True),
    early_stopping,
    reduce_lr]

# save hyperparameters in a JSON file
model.save(os.path.join(OUT_PATH,f"{model_name}_architecture"), save_weights = False)

# time the execution of the model training
start_time = time.time()
# train model
history = model.model.fit(x=trn_generator, validation_data = val_generator, 
                        epochs = N_EPOCHS, callbacks=model_checkpoint_callback)
print("Training took", time.time() - start_time, "seconds")

loss = history.history['loss']
print(loss)
val_loss = history.history['val_loss']
print(val_loss)



fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
ax[0].plot(loss)
ax[1].plot(val_loss)
ax[0].set_title("Training Loss")
ax[1].set_title("Validation Loss")
plt.xlabel("number of epochs")
fig.text(-0.001, 0.5, "loss (MAE for artifically gapped pixels)", va='center', rotation='vertical')
plt.suptitle("Loss development")
plt.show()

# save the figure
fig.savefig(f"./loss_model_pics/{model_name}.png", dpi=300)

#%%
plt.figure()
plt.plot(loss, label='loss')
plt.plot(val_loss, label='val_loss')
plt.legend()
plt.ylim(0,1)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and validation loss')
plt.show()

# save the figure
fig.savefig(f"./loss_model_pics/{model_name}_zoom.png", dpi=300)

# %%
