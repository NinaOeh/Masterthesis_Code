from lib.pDINCAEModel import DINpCAE
from lib.pDataGenerator import DataGenerator
import tensorflow as tf
import os
import shutil
import time
from tensorflow.python.framework.ops import disable_eager_execution
import h5py
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np


print("Starting the training")
disable_eager_execution()

def identity(x):
    return x

EPOCHS = 500
DATA_PATH_TRAINING = "../../CHL_DATA/InputData_RIGA/TrainData_NC_RIGA"
DATA_PATH_VALIDATION = "../../CHL_DATA/InputData_RIGA/ValData_NC_RIGA"
DATA_PATH_IN_TEMP = "../../CHL_DATA/InputData_RIGA/"
BATCH_SIZE = 10

varname = "CHL"
outdir = "../data/models_pDINCAE" 
jitter_std = 0.05
learning_rate = 0.000153 #135
ntime_win = 5
nvar = 6 + 2*ntime_win - 2 + 1
transfun = (identity, identity)
shuffle_buffer_size = 3*15
dim = (20,160,160)
model_name = "pDINCAE_1_1_1_0_1_0_2_A_T_S_S_dataperc30-100"


# first letter: ratio alpha (0 - 0.1, 2 - 0.2, 1 - 0.3, 3 - 0.4)
# secondletter: number of layers (0 - 4, 1 - 5, 2 - 6)
# third letter: batch size (0 - 5, 1 - 10, 2 - 20, 3 - 8)
# forth letter: number of epochs (0 - 500, 1 - 1000, 2 - 2000)
# fifth letter: learning rate (0 - 1e-3, 1 - 1e-4, 2 - 1e-5, 3 - decaying, 4 - plateau)
# sixth letter: number of time windows (0 - 5, 1 - 10, 2 - 15)
# seventh letter: data preprocessing (0 - normal scale * 10, 1 - log(log(+1)+1)*10, 2 - (log) * 10), 3 - x^(-0.14) * mean/std * 10 (boxcox)
#                                     4 - boxcox * 10 (boxcox), 5 - log(log())*10, 6 - dagum, 7 - normalixed boxcox, 8 normalized
# eigth letter: regularixation (0.1,0.1 ; 0.1,0.2; A: custom1)
# nineth letter: concat intermediate output with input (T-True,F(None)-False)
# tenth letter: P - pooling, S - strided conv
# eleventh letter: skip connection, S - sum, C - concat


# Get the temporal mean of the entire dataset
data_olci = xr.open_dataset(os.path.join(DATA_PATH_IN_TEMP,f'input_data_olci.nc'))
meandata_olci = np.nanmean(data_olci['CHL'], axis=0,keepdims=True)
data_multi = xr.open_dataset(os.path.join(DATA_PATH_IN_TEMP,f'input_data_multi.nc'))
meandata_multi = np.nanmean(data_multi['CHL'], axis=0,keepdims=True)


train_datagen = DataGenerator(DATA_PATH_TRAINING, batch_size = BATCH_SIZE, 
                              ntime_win=ntime_win, cb=True, mean_olci=None,
                              mean_multi=None)
val_datagen = DataGenerator(DATA_PATH_TRAINING, batch_size = BATCH_SIZE, 
                            ntime_win=ntime_win, cb=True, mean_olci=None,
                              mean_multi=None)


model = DINpCAE(nx=dim[1], ny=dim[2], ntime=ntime_win,
                learning_rate=learning_rate, 
                enc_nfilter_internal=[16, 30, 58, 110, 209],
                alpha=0.3, nvar=nvar,
                skip="Sum", stride_param=1, l1=0.05 ,
                l2=0.035, concat="True" )
model.summary()

# save model at the end of each epoch
checkpoint_dir = os.path.join(outdir, model_name) 

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=100)

# Reduce learning rate when a metric has stopped improving.
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", patience=50)


class PrintLR(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    print('\nLearning rate for epoch {} is {}'.format(epoch + 1, model.learning_rate))

if os.path.exists(checkpoint_dir):
    shutil.rmtree(checkpoint_dir)
os.mkdir(checkpoint_dir)
checkpoint_filepath = os.path.join(checkpoint_dir, "epoch_{epoch:02d}.h5")
model_checkpoint_callback = [tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filepath,
    save_weights_only = True,
    #save_freq = "epoch",
    #period = 100,
    save_best_only=True),
    PrintLR(),
    early_stopping,
    reduce_lr
    ]

# save hyperparameters in a JSON file
model.save(os.path.join(outdir,f"{model_name}_architecture"), save_weights = False)


# time the execution of the model training
start_time = time.time()
# train model
history = model.model.fit(x=train_datagen, validation_data = val_datagen, 
                        epochs = EPOCHS, callbacks=model_checkpoint_callback)
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
fig.savefig(f"{model_name}.png", dpi=300)

plt.figure()
plt.plot(loss, label='loss')
plt.plot(val_loss, label='val_loss')
plt.legend()
plt.ylim(0,10)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and validation loss')
plt.show()

# save the figure
fig.savefig(f"{model_name}_zoom.png", dpi=300)
# %%
