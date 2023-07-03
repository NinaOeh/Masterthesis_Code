#%%
#%load_ext tensorboard
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

import io
import imageio
from IPython.display import Image, display
from ipywidgets import widgets, Layout, HBox
import os
from netCDF4 import Dataset, num2date
from lib.DataLoader import Data_Loader, Data_Loader_Features
from lib.CHL_LSTM import model_creator
from tensorflow.python.framework.ops import disable_eager_execution
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
np.random.seed(42)
from datetime import datetime
import keras.backend as K
K.clear_session()
np.random.seed(42)
from numpy.random import seed
seed(2)
tf.keras.utils.set_random_seed(2)
tf.config.experimental.enable_op_determinism()

print("Starting the training")
disable_eager_execution()

gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        # # Currently, memory growth needs to be the same across GPUs
        # for gpu in gpus:
        #     tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Prepare CHL predicted data to be passed to LSTM model
DATA_PATH_CHL = "../CHL_DATA/InputData_KTT/input_data_olci_interpolated_pCONV3D_FINAL.nc"
DATA_PATH_CHL_INIT = "../CHL_DATA/InputData_KTT/input_data_olci.nc"
DATA_PATH_Features = f"../CHL_DATA/InputData_KTT/Features"


MODELNAME = "model_3_2_1_1_1_1_KTT_group8_missing"

# model naming convention:
# First Number: Learning rate (0-0.001, 1-0.0001, 2-0.00001, 3-custom)
# Second Number: Batch size (0-1, 1-2, 2-4, 3-8, 4-16)
# Third Number: Training generation (data assimilation, rolling window) (0 - no DA, 1 - DA)
# Forth number: Time range (0-2, 1-5, 2-7, 3-10)
# Fifth number: Features added (0 - no features, 1 - features added, 2 - features missing)
# Sixth number: Chl- data transformation (0 - min-max, 1 - log, 2 - log+min-max)
# sevent_letter better data processing and splitting (N - no, Y - yes)


# Define modifiable training hyperparameters.
epochs = 100
batch_size = 4 # 4
learning_rate = 0.0002579




dim=(20,160,160)

#interpolated_value = 'CHL_interpolated_nn'
interpolated_value = 'CHL_interpolated_pCONV3D'

dataset, landmask, _, _ = Data_Loader_Features(DATA_PATH_CHL, 
                                               DATA_PATH_Features, 
                                               interpolated_value, 
                                               dim=dim, 
                                               datatrans='log', feat='all')



# In the first step we will split the data in training and remaining dataset
X_train_ind, X_rem_ind, y_train, y_rem = train_test_split(np.arange(dataset.shape[0]),
                                                          np.arange(dataset.shape[0]), 
                                                          train_size=0.8,
                                                          random_state=42)

# Now since we want the valid and test size to be equal (10% each of overall data). 
# we have to define valid_size=0.5 (that is 50% of remaining data)
X_valid_ind, X_test_ind, y_valid, y_test = train_test_split(X_rem_ind,y_rem, 
                                                            test_size=0.5,
                                                            random_state=42)
    
train_dataset = dataset[X_train_ind]
test_dataset = dataset[X_test_ind]
val_dataset = dataset[X_valid_ind]

print(f"Train dataset shape: {train_dataset.shape}")
print(f"Test dataset shape: {test_dataset.shape}")
print(f"Validation dataset shape: {val_dataset.shape}")

#%%

# Create overlapping windows of history and target data.      
def create_overlapping_frames(dataset, history_size=10,
                      target_size=5):
    sz = dataset.shape
    X = np.empty((0,history_size,sz[2],sz[3], sz[4]-2))
    y = np.empty((0,target_size,sz[2],sz[3]))
    start_index = 0
    start_index = start_index + history_size
    end_index = dataset.shape[1] - target_size +1

    for i in range(start_index, end_index):
        # omit the chl feature
        print(i-history_size)
        d_temp = dataset[:,i-history_size:i,:,:,0].shape
        X_temp = np.empty((d_temp[0], d_temp[1], d_temp[2], d_temp[3], sz[4]-2))
        print(X_temp.shape)
        X_temp[:,:,:,:,:3] = dataset[:,i-history_size:i,:,:,:3]
        # X_temp[:,:,:,:,4] = dataset[:,i-history_size:i,:,:,5]
        X_temp[:,:,:,:,3:8] = dataset[:,i-history_size:i,:,:,4:9]
        X_temp[:,:,:,:,8:] = dataset[:,i-history_size:i,:,:,10:]
        X = np.append(X, X_temp, axis=0)
        y = np.append(y, dataset[:,i:i+target_size,:,:,0], axis=0)
        
    # expand dims to add channel dimension
    y = np.expand_dims(y, axis=4)   
    # swap column 1 and 4 of y
    y = np.swapaxes(y, 1, 4)

    return X, y

# Create shifted frames of the data.
# `x` is frames 0 to n - 1, and `y` is frames 1 to n.
def create_shifted_frames(data, days):
    # x must have shape (num_samples, time_steps, height, width, channels)
    x = data[:, 0 : data.shape[1] - days, :, :]
    y = data[:, days : data.shape[1], :, :, 0]
    
    # expand dims to add channel dimension
    y = np.expand_dims(y, axis=4)
    
    # y must have shape (num_samples, output_height, output_width, output_channels)
    # the number of days to be predicted are the output channels
    # swap column 1 and 4 of y
    y = np.swapaxes(y, 1, 4)

    print(f"shape of target data: {y.shape}")
    return x, y


# Apply the processing function to the datasets.
# x_train, y_train = create_shifted_frames(train_dataset, 10)
# x_val, y_val = create_shifted_frames(val_dataset, 10)
# x_test, y_test = create_shifted_frames(val_dataset, 10)

# print(f"shape of training data: {x_train.shape}")
# print(f"shape of training labels: {y_train.shape}")
# print(f"shape of validation data: {x_val.shape}")
# print(f"shape of validation labels: {y_val.shape}")

# Create overlapping windows of history and target data.
x_train, y_train = create_overlapping_frames(train_dataset, 10)
x_val, y_val = create_overlapping_frames(val_dataset, 10)
x_test, y_test = create_overlapping_frames(val_dataset, 10)

print(f"shape of training data: {x_train.shape}")
print(f"shape of training labels: {y_train.shape}")
print(f"shape of validation data: {x_val.shape}")
print(f"shape of validation labels: {y_val.shape}")

#%%

# fill nan values with 0
x_train = np.nan_to_num(x_train)
y_train = np.nan_to_num(y_train)
x_val = np.nan_to_num(x_val)
y_val = np.nan_to_num(y_val) 
x_test = np.nan_to_num(x_test)
y_test = np.nan_to_num(y_test)


# Inspect the dataset.
print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))
print("Test Dataset Shapes: " + str(x_test.shape) + ", " + str(y_test.shape))

# Define some callbacks to improve training.
early_stopping = keras.callbacks.EarlyStopping(monitor="loss", patience=20)

# Reduce learning rate when a metric has stopped improving.
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="loss", patience=10)

# create model
model = model_creator(x_train.shape[2:],
                    learning_rate=learning_rate,
                    dropout_rate=0,  #0.1,
                    filter=2, #2,
                    kernel_size_1=7, #7,
                    kernel_size_2=5, #5,
                    kernel_size_3=7, #7,
                    act="LeakyReLU", #"LeakyReLU",
                    kernel_size_final=5) #5)


print(x_train.shape)
print(x_train.shape)
print(y_train[0].shape)
print(len(y_train))


print(model.summary())

# log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Fit the model to the training data.
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping, reduce_lr], #, tensorboard_callback],
)

model.save(f'{MODELNAME}.h5')



loss = history.history['loss']
print(loss)
val_loss = history.history['val_loss']
print(val_loss)



#losses = [0.021444928184969396, 0.021829163769214618, 0.015117645059235213, 0.011016613630600637, 0.01127133757818672, 0.010349192606177317, 0.008785572652591438, 0.009438856127217593, 0.008965359347668, 0.00868611183108353, 0.008855007336156942, 0.00891577153716509, 0.008403892193000945, 0.007786270991967219, 0.007527615558412835, 0.007517810795234698, 0.007603797557616107, 0.008185576720634566, 0.008037153835175559, 0.007869593696895896, 0.007936046871913188, 0.007943642550721629, 0.00757788933058292, 0.00788929900608775, 0.007580976118333638, 0.008191737777939656, 0.007966175652676967, 0.008025781328153864, 0.00846280523280545, 0.009189790035967102, 0.008137481160058708, 0.007502571885508099, 0.007358306506719645, 0.007488784738021289, 0.007361174224614643, 0.007177630079506964, 0.00728643021302116, 0.007380403799754454, 0.007904022020605824, 0.007345909583552673, 0.00816561150216929, 0.007853089955283284, 0.00794754221058655, 0.007352833126433103, 0.007625990470576032, 0.007740247253578429, 0.007577101681551828, 0.007672590283086387, 0.00739338712641833, 0.00771620927276317]
#val_losses = [0.010789214487886056, 0.014338724222034216, 0.009014991950243712, 0.006934920558705926, 0.010725156287662685, 0.0071222806291189045, 0.006143609172431752, 0.006484717025887221, 0.007713395892642439, 0.00792038645595312, 0.006634794018464163, 0.005976196736446582, 0.005506358155980706, 0.006304502789862454, 0.00568677675910294, 0.005752452957676723, 0.0055610362411243845, 0.006762572133447975, 0.0071278362767770885, 0.006810366851277649, 0.006219798047095537, 0.005790111480746418, 0.006871153018437326, 0.005382029816973955, 0.005496995369321666, 0.006301868171431124, 0.005739780177827924, 0.006549380673095584, 0.0062521673971787095, 0.006200107024051249, 0.005374977923929691, 0.00541409042198211, 0.006083521875552833, 0.005260852808714844, 0.005455459852237254, 0.0052995393751189114, 0.005825907795224339, 0.006072636775206775, 0.0048968186369165775, 0.005579903171746991, 0.006799104996025563, 0.0061572642007377, 0.005616471989196725, 0.006120622542221099, 0.006378684367518872, 0.005994013976305723, 0.00574103023391217, 0.005313764838501811, 0.005614107777364552, 0.005370854030479677]
#loss = [0.0220371031645322, 0.02021764243509984, 0.018627824024694813, 0.017841922397520846, 0.018460884979903336, 0.01754837954365762, 0.016809867493376682, 0.01667364455458549, 0.01665896422899805, 0.016324842325979617]
#val_loss = [0.02006171699613333, 0.011622374829312321, 0.013391062431037427, 0.013923266367055476, 0.012772558780852705, 0.013398197176866234, 0.013309618597850204, 0.011463922448456287, 0.010867246799170971, 0.01221849232679233]
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
fig.savefig(f"./{MODELNAME}.png", dpi=300)

# %%
import matplotlib.pyplot as plt

plt.figure()
plt.plot(loss, label='loss')
plt.plot(val_loss, label='val_loss')
plt.legend()
plt.ylim(0,0.5)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and validation loss')
plt.show()
plt.savefig(f"./{MODELNAME}_zoom.png", dpi=300)
# %%


# %%
# following tutorial to show the model features
# https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/


model_name = "Viz_model"
# else try to load model like this:
model = load_model(f"../data/models_DINCAE/{model_name}/epoch_01.h5", 
                   custom_objects={'loss_total': DINCAE.loss_total, 'rmse_loss': DINCAE.loss_mae})



# show the model architecture
# summarize feature map shapes
for i in range(len(model.layers)):
    layer = model.layers[i]
    # check for convolutional layer
    if 'conv' not in layer.name:
        continue
    # summarize output shape
    print(i, layer.name, layer.output.shape)
    
#%%
# visualize the model architecture
import visualkeras

visualkeras.layered_view(model, legend=True, scale_xy=3, scale_z=0.1, to_file=f"../data/models_DINCAE/{model_name}/model_architecture.png").show()


#%%
# redefine model to output right after the first hidden layer
model = Model(inputs=model.inputs, outputs=model.layers[1].output)
# load a single file 
X_test, lon, lat, time_, meandata = Test_Data_Loader(files[0], dim=(20,160,160), ntime_win=3, obs_err_std = [1.])
    
X_single = X_test[x,:,:,:]
X_single = np.expand_dims(X_single, axis=0)
print(X_single.shape)
pred = model.predict(X_single)

# get feature map for first hidden layer
feature_maps = model.predict(pred)