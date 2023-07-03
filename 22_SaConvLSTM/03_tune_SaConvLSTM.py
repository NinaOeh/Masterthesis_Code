#%%
# @misc{omalley2019kerastuner,
#     title        = {KerasTuner},
#     author       = {O'Malley, Tom and Bursztein, Elie and Long, James and Chollet, Fran\c{c}ois and Jin, Haifeng and Invernizzi, Luca and others},
#     year         = 2019,
#     howpublished = {\url{https://github.com/keras-team/keras-tuner}}
# }

import kerastuner as kt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from lib.DataLoader import Data_Loader, Data_Loader_Features
from lib.SaConvLSTM_Model import Sa_build_model
from tensorflow.python.framework.ops import disable_eager_execution
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
np.random.seed(42)

print("Starting the training")
disable_eager_execution()

gpus = tf.config.list_physical_devices('GPU')
print(f"Tensorflow version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

if gpus:
    try:
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Prepare CHL predicted data to be passed to LSTM model
DATA_PATH_CHL = "../CHL_DATA/InputData_RIGA/input_data_olci_interpolated_pCONV3D.nc"
DATA_PATH_CHL_INIT = "../CHL_DATA/InputData_RIGA/input_data_olci.nc"
DATA_PATH_Features = f"../CHL_DATA/InputData_RIGA/Features"


# Define modifiable training hyperparameters.
epochs = 5000
batch_size = 2
learning_rate = 0.001


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


# Create overlapping windows of history and target data.      
def create_overlapping_frames(dataset, history_size=10,
                      target_size=5):
    sz = dataset.shape
    X = np.empty((0,history_size,sz[2],sz[3], sz[4]))
    y = np.empty((0,target_size,sz[2],sz[3]))
    start_index = 0
    start_index = start_index + history_size
    end_index = dataset.shape[1] - target_size

    for i in range(start_index, end_index):
        X = np.append(X, dataset[:,i-history_size:i,:,:, :], axis=0)
        y = np.append(y, dataset[:,i:i+target_size,:,:,0], axis=0)
        
    # expand dims to add channel dimension
    y = np.expand_dims(y, axis=4)   
    # swap column 1 and 4 of y
    y = np.swapaxes(y, 1, 4)

    return X, y

# Create overlapping windows of history and target data.
x_train, y_train = create_overlapping_frames(train_dataset, 10)
x_val, y_val = create_overlapping_frames(val_dataset, 10)
x_test, y_test = create_overlapping_frames(val_dataset, 10)


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

#%%

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        model = Sa_build_model(x_train.shape[2:],
                    learning_rate=hp.Float("learning_rate", min_value=1e-5, 
                                        max_value=1e-2, sampling="log"),
                    dropout_rate=0, #hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1),
                    filter=0, #hp.Int("filter", min_value=0, max_value=3, step=1),
                    kernel_size_1=5, #hp.Int("kernel_size_1", min_value=3, max_value=7, step=2),
                    kernel_size_2=5, #hp.Int("kernel_size_2", min_value=3, max_value=7, step=2),
                    kernel_size_3=7, #hp.Int("kernel_size_3", min_value=3, max_value=7, step=2),
                    act="LeakyReLU",#hp.Choice("act", ["LeakyReLU", "tanh"]),
                    kernel_size_final=5, #hp.Int("kernel_size_final", min_value=3, max_value=7, step=2),
                    )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=2, #hp.Choice("batch_size", [2,4,8]),
            **kwargs,
        )
        
tuner = kt.BayesianOptimization(
    MyHyperModel(),
    objective="val_loss",
    max_trials=100,
    overwrite=True,
    directory="tuning_models_SaConvLSTM",
    project_name="tune_SaConvLSTM_1",
)
        
    
cp_callback = [tf.keras.callbacks.ModelCheckpoint(filepath="./tune_SaConvLSTM",
                                                 save_best_only=True),
               tf.keras.callbacks.EarlyStopping('val_loss', patience=20)]


print("The search begins...")
tuner.search(x_train,y_train,
             validation_data=(x_val, y_val),
             epochs=50,
             callbacks=cp_callback)

best_model = tuner.get_best_models(1)[0]

best_hyperparameters = tuner.get_best_hyperparameters(1)[0]

print(best_hyperparameters)



# %%
