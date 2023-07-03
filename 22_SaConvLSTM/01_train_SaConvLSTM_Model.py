#%%
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

from IPython.display import Image, display
from ipywidgets import widgets, Layout, HBox
import os
from netCDF4 import Dataset, num2date
from lib.DataLoader import Data_Loader, Data_Loader_Features
from lib.SaConvLSTM_Model import Sa_build_model
from tensorflow.python.framework.ops import disable_eager_execution
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
DATA_PATH_CHL = "../CHL_DATA/InputData_RIGA/input_data_olci_interpolated_pCONV3D_FINAL.nc"
DATA_PATH_CHL_INIT = "../CHL_DATA/InputData_RIGA/input_data_olci.nc"
DATA_PATH_Features = f"../CHL_DATA/InputData_RIGA/Features"


MODELNAME = "SaConvLSTM_3_1_1_1_1_0_RIGA_optimized"

# model naming convention:
# First Number: Learning rate (0-0.001, 1-0.0001, 2-0.00001, 3-custom)
# Second Number: Batch size (0-1, 1-2, 2-4, 3-8, 4-16)
# Third Number: Training generation (data assimilation, rolling window) (0 - no DA, 1 - DA)
# Forth number: Time range (0-2, 1-5, 2-7, 3-10)
# Fifth number: Features added (0 - no features, 1 - features added, 2 - features missing)
# Sixth number: Chl- data transformation (0 - min-max, 1 - log, 2 - log+min-max)

# Define modifiable training hyperparameters.
epochs = 100
batch_size = 2 #2
learning_rate = 9.2709e-05 
dropout_rate=0
filter=0
kernel_size_1=5
kernel_size_2=5
kernel_size_3=7
act="LeakyReLU"
kernel_size_final=5


dim=(20,160,160)

#interpolated_value = 'CHL_interpolated_nn'
interpolated_value = 'CHL_interpolated_pCONV3D'

dataset, landmask, _, _ = Data_Loader_Features(DATA_PATH_CHL, 
                                               DATA_PATH_Features, 
                                               interpolated_value, 
                                               dim=dim, 
                                               datatrans='minmax', feat='all')


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
    X = np.empty((0,history_size,sz[2],sz[3], sz[4]))
    y = np.empty((0,target_size,sz[2],sz[3]))
    start_index = 0
    start_index = start_index + history_size
    end_index = dataset.shape[1] - target_size +1

    for i in range(start_index, end_index):
        # omit the chl feature
        print(i-history_size)
        d_temp = dataset[:,i-history_size:i,:,:,0].shape
        X_temp = np.empty((d_temp[0], d_temp[1], d_temp[2], d_temp[3], sz[4]))
        print(X_temp.shape)
        X_temp[:,:,:,:,:] = dataset[:,i-history_size:i,:,:,:]
        # X_temp[:,:,:,:,3:8] = dataset[:,i-history_size:i,:,:,4:9]
        # X_temp[:,:,:,:,8:] = dataset[:,i-history_size:i,:,:,10:]
        X = np.append(X, X_temp, axis=0)
        y = np.append(y, dataset[:,i:i+target_size,:,:,0], axis=0)
        
    # expand dims to add channel dimension
    y = np.expand_dims(y, axis=4)   
    # swap column 1 and 4 of y
    y = np.swapaxes(y, 1, 4)
    
    # y = [y[:,:,:,:,i] for i in range(y.shape[-1])]
    # y = [np.expand_dims(i, axis=4) for i in y]

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

#%%

print(f"shape of training data: {x_train.shape}")
print(f"shape of training labels: {y_train.shape}")
print(f"shape of validation data: {x_val.shape}")
print(f"shape of validation labels: {y_val.shape}")
# fill nan values with 0
x_train = np.nan_to_num(x_train)
y_train = np.nan_to_num(y_train) #[np.nan_to_num(y) for y in y_train]
x_val = np.nan_to_num(x_val)
y_val = np.nan_to_num(y_val) #[np.nan_to_num(y) for y in y_val]  
x_test = np.nan_to_num(x_test)
y_test = np.nan_to_num(y_test) #[np.nan_to_num(y) for y in y_test]

# Inspect the dataset.
print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
print("Test Dataset Shapes: " + str(x_test.shape) + ", " + str(y_test.shape))
print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))

# Define some callbacks to improve training.
early_stopping = keras.callbacks.EarlyStopping(monitor="loss", patience=20)

# Reduce learning rate when a metric has stopped improving.
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="loss", patience=10)

# create model
model = Sa_build_model(x_train.shape[2:], 
                       learning_rate=learning_rate,
                       dropout_rate=dropout_rate,
                       filter=filter,
                       kernel_size_1=kernel_size_1,
                       kernel_size_2=kernel_size_2,
                       kernel_size_3=kernel_size_3,
                       act=act,
                       kernel_size_final=kernel_size_final)

# session = keras.backend.get_session()
# init = tf.global_variables_initializer()
# session.run(init)

# log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

print(x_train.shape)
print(y_train.shape)

print(model.summary())

# Fit the model to the training data.
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=2,
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

#%%
# import matplotlib.pyplot as plt

# loss = [0.0015445898078814116, 0.0009875858356223886, 0.0009057165101497761, 0.000855102750204235, 0.0008341481689959428, 0.0008307481795520027, 0.0008159384932704606, 0.0007946343401862258, 0.0007674678051550579, 0.0007776214776829154, 0.0007555158150713756, 0.0007516404654230788, 0.0007329516344586269, 0.0007369711665805798, 0.0007221369759225798, 0.0006989728387231259, 0.0007101377359012495, 0.0007133255639272902, 0.0007052426417840661, 0.0006916784610062301, 0.0006925794652554097, 0.0006961350844174556, 0.0006501989967193924, 0.0006467870883391919, 0.0006407073794377481, 0.0006410945689322443, 0.0006414947641682014, 0.0006388665637345397, 0.000635085532521443, 0.0006348621884155347, 0.0006334111201975427, 0.0006326804467716027, 0.0006319055675113624, 0.0006268707849714603, 0.0006275313443856667, 0.0006257113887873889, 0.0006256488105701703, 0.0006260478516163825, 0.0006263805857465208, 0.0006235782940649671, 0.0006253947316394146, 0.0006226214023332137, 0.0006240621772343042, 0.0006255333923822822, 0.0006249519925027549, 0.0006220684462040243, 0.0006240385811071139, 0.0006245573416775982, 0.0006248617566008171, 0.0006241464501284567, 0.0006234188687878552, 0.0006235023272991482, 0.0006242673820501344, 0.0006249279619156811, 0.0006235992831706598, 0.0006239093700633594, 0.000626393339516981, 0.0006250906486571808, 0.0006235131548417665, 0.000622033144699762, 0.0006244953589090188, 0.0006248150165045952, 0.0006214029161964316, 0.0006233560564290094, 0.0006250822161500519, 0.000624186109210289, 0.0006220749404222646, 0.000624038313404313, 0.0006226141664916901, 0.0006240464601170708, 0.0006236353961736182, 0.0006242120176499189, 0.0006232295992368883, 0.0006229241605667683, 0.0006222367608659641, 0.0006241297217186641, 0.0006243568183833246, 0.000623891547244663, 0.0006240863011633927, 0.0006230937331172361, 0.0006231514381471317, 0.0006228469344703397, 0.0006238880340714712]
# val_loss = [0.001798652116364489, 0.0017125222482718527, 0.0015534188986445466, 0.0013867473462596536, 0.0011855181151380141, 0.0009143702870157236, 0.0009546737865700077, 0.0014281531718249123, 0.0007365462137386203, 0.0007331608115540197, 0.0007791089165645342, 0.0008897105096063266, 0.0007260078350858142, 0.002507310570217669, 0.0007676929410081357, 0.0008000268309842796, 0.0016017203762506446, 0.0019340952392667532, 0.0011308322369586675, 0.0011236078629735858, 0.0007831312968240429, 0.0008181985564685116, 0.0007187236381772285, 0.0007018438685918227, 0.000701433327049017, 0.0007121816655853763, 0.0007221975324985881, 0.0007129134697606787, 0.0007000652316492051, 0.0007114881979456792, 0.0006931745447218418, 0.0007048731475758056, 0.0006947454618057236, 0.0007036466442514211, 0.0006988379949082931, 0.000698505574837327, 0.0006984373855326945, 0.0006991915133160849, 0.0006982296200779577, 0.0006970259234852468, 0.00070254555127273, 0.0007021818115996817, 0.0006952859073256453, 0.0006959342184321334, 0.0006960936104102681, 0.0006965889401423435, 0.0006964040163438767, 0.0006967312539927662, 0.0006966909432473283, 0.0006966266897507012, 0.0006967133706590781, 0.000696268219811221, 0.0006969340737365807, 0.0006966674108601485, 0.0006965613652331133, 0.0006965814439657455, 0.0006965186534216627, 0.0006965307499437282, 0.0006966673642940199, 0.0006965988082811237, 0.0006966392402925219, 0.0006965960201341659, 0.0006966147610607247, 0.0006965903342158223, 0.0006966822829175119, 0.000696611295764645, 0.0006966832685672367, 0.0006967000294631968, 0.0006966834674434116, 0.0006966551047905038, 0.0006967575240802641, 0.0006965965944497536, 0.0006966207719718416, 0.0006965311292636519, 0.0006965476107628396, 0.0006967040807163964, 0.0006968001078348606, 0.0006966252665733918, 0.0006967264702931668, 0.0006967011742138614, 0.0006966634042328224, 0.0006965967962363114, 0.0006966112511387715]


plt.figure()
plt.plot(loss, label='loss')
plt.plot(val_loss, label='val_loss')
plt.legend()
plt.ylim(0,0.0023)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and validation loss')
plt.show()
plt.savefig(f"./{MODELNAME}_zoom.png", dpi=300)


#%%
import scipy
import numpy as np
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    print(n)
    m, se = np.nanmean(a), scipy.stats.sem(a, nan_policy='omit')
    h = se * scipy.stats.t.ppf((1+confidence)/2., n-1)
    return m, h

rmses= [0.6284018278145891, 0.5355297077219788, 0.4854439969696279, 0.640690053193422,
        0.3096614256600618, 0.46226074277533086, 0.5200567610101208, 0.4786895323903165, 
        0.9615345426337657, 0.6547352038222176, 1.0099874818362005, 0.9649475139074197, 
        0.4873387022021955, 0.5621329977655503]
r2s = [0.5549605435839619, 0.5599532452168646, 0.48659218879320587, 0.27585207502503606,
       0.37458638653165005, 0.2840746102764375, 0.5786975134973905, 0.5218819377698368,
       0.40804498553623253, 0.6730264224929403, 0.4910792134310727, 0.6053748136521113,
       0.5447525675044379, 0.43972358651796734]

print(mean_confidence_interval(rmses))
print(mean_confidence_interval(r2s))

# %%
