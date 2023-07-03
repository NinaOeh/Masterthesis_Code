#%%
# @misc{omalley2019kerastuner,
#     title        = {KerasTuner},
#     author       = {O'Malley, Tom and Bursztein, Elie and Long, James and Chollet, Fran\c{c}ois and Jin, Haifeng and Invernizzi, Luca and others},
#     year         = 2019,
#     howpublished = {\url{https://github.com/keras-team/keras-tuner}}
# }

import kerastuner as kt
from netCDF4 import Dataset, num2date

# Import modules from lib/ directory
from lib.DINCAEModel import DINCAE
from lib.DataGenerator import DataGenerator, Test_Data_Loader
import time
from keras.models import load_model, Model
import tensorflow as tf
import os
from typing import List

print(f"Tensorflow version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


EPOCHS = 1000
DATA_PATH_TRAINING = "../../CHL_DATA/train_data_nc"
DATA_PATH_VALIDATION = "../../CHL_DATA/valid_data_nc"
BATCH_SIZE = 10

def identity(x):
    return x

ntime_win = 5
nvar = 6 + 2*ntime_win - 2
transfun = (identity, identity)
dim = (20,160,160)

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync

train_datagen = DataGenerator(DATA_PATH_TRAINING, batch_size = BATCH_SIZE, ntime_win=ntime_win)
val_datagen = DataGenerator(DATA_PATH_TRAINING, batch_size = BATCH_SIZE, ntime_win=ntime_win)

print(train_datagen)


def model_builder(hp):
    #with strategy.scope():
        model = DINCAE(nx=dim[1], ny=dim[2], ntime=ntime_win,
                    learning_rate=hp.Float("learning_rate", min_value=1e-5, 
                                        max_value=1e-2, sampling="log"),
                    alpha=hp.Choice("alpha1", [0.1, 0.3, 0.5, 0.7, 0.9]),
                    nvar=nvar,
                    )
        return model.model
    
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="./tuneDINCAE",
                                                 save_best_only=True)

tuner = kt.BayesianOptimization(
    model_builder,
    objective=kt.Objective("loss", direction="min") ,
    max_trials=500,
    project_name="DINCAE_tune") #,distribution_strategy=strategy


print("Model loading completed")
tuner.search(train_datagen,
             validation_data=val_datagen,
             epochs=50,
             callbacks=[cp_callback])

best_model = tuner.get_best_models(1)[0]

best_hyperparameters = tuner.get_best_hyperparameters(1)[0]

print(best_hyperparameters)



# %%
