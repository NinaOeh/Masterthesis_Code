'''
Script to predict the missing data in the input data files using the DINCAE model

'''

#%%
%matplotlib ipympl
import os
import sys
import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt

from skimage import io as skio
from skimage import util as skutil

from netCDF4 import Dataset, num2date

# Import modules from lib/ directory
from lib.DINCAEModel import DINCAE
from lib.DataGenerator import DataGenerator, Test_Data_Loader
import time
from keras.models import load_model, Model
import tensorflow as tf
import xarray as xr
np.random.seed(42)

#%%
print("Using TensorFlow version", tf.__version__)

model_name = "DINCAE_1_1_0_0_0_0_1_0_RIGA_dataperc20-100"
DATA_PATH_IN = "../../CHL_DATA/InputData_RIGA/TestData_NC_RIGA"
DATA_PATH_IN_TEMP = "../../CHL_DATA/InputData_RIGA/"
DATA_PATH_OUT = f"../data/prediction_{model_name}_GapSizeVar"
#%%

if os.path.exists(DATA_PATH_OUT):
    if len(os.listdir(DATA_PATH_OUT)) > 0:
        sys.exit("Output directory exists and is not empty")
else:
    os.makedirs(DATA_PATH_OUT)

#%%
model = DINCAE.load(f"../data/models_DINCAE/{model_name}_architecture", 
                    weights_name = f"../data/models_DINCAE/{model_name}/epoch_401.h5")
    
#%%
def savesample(fname,m_rec,σ2_rec,lon,lat, time_, meandata, y=False, x=False, mask=None):
    fill_value = -9999.
    # todo apply transfun to sigma_rec

    sigma_rec = np.sqrt(np.abs(σ2_rec))
    mask = np.where(mask == 0, 1, 0)
    
    if y == True:
        recdata = np.where(m_rec == fill_value, np.nan, m_rec) 
        recdata = np.where(recdata == 0, np.nan, recdata) 
    elif x == True:
        recdata = np.where(m_rec == 0, np.nan, m_rec)
    else:
        recdata = m_rec
        
    recdata = recdata  + meandata    

    # create file
    root_grp = Dataset(fname, 'w', format='NETCDF4')

    # dimensions
    root_grp.createDimension('time', None)
    root_grp.createDimension('lon', len(lon))
    root_grp.createDimension('lat', len(lat))

    # variables
    nc_lon = root_grp.createVariable('lon', 'f4', ('lon',))
    nc_lat = root_grp.createVariable('lat', 'f4', ('lat',))
    nc_meandata = root_grp.createVariable(
        'meandata', 'f4', ('lat','lon'),
        fill_value=fill_value)

    nc_mean_rec = root_grp.createVariable(
        'mean_rec', 'f4', ('time', 'lat', 'lon'),
        fill_value=fill_value)

    nc_sigma_rec = root_grp.createVariable(
        'sigma_rec', 'f4', ('time', 'lat', 'lon',),
        fill_value=fill_value)

    # data
    nc_lon[:] = lon
    nc_lat[:] = lat
    nc_meandata[:,:] = meandata
    print(m_rec.shape)
    for n in range(m_rec.shape[0]):
        nc_mean_rec[n,:,:] = np.ma.masked_array(
            recdata[n,:,:],mask)
        nc_sigma_rec[n,:,:] = np.ma.masked_array(
            sigma_rec[n,:,:],mask)

    root_grp.close()
    
#%%
# Get the temporal mean of the entire dataset
data_olci = xr.open_dataset(os.path.join(DATA_PATH_IN_TEMP,f'input_data_olci.nc'))
meandata_olci = np.nanmean(data_olci['CHL'], axis=0,keepdims=True)
data_multi = xr.open_dataset(os.path.join(DATA_PATH_IN_TEMP,f'input_data_multi.nc'))
meandata_multi = np.nanmean(data_multi['CHL'], axis=0,keepdims=True)

#%%
    
files = sorted(
            [
                os.path.join(DATA_PATH_IN, fname)
                for fname in os.listdir(DATA_PATH_IN)
                if fname.endswith(".nc") and fname.startswith("file_") 
                and "00-10" not in fname
                and "10-20" not in fname
                and "50-100" in fname
            ])


with open(os.path.join(DATA_PATH_OUT,f'Prediction_times.txt'), 'w') as f:
        f.write("Times: ")
start = time.process_time()

# generate an array as seeds for the perlin noise generator
rand_seeds = np.arange(0,20*len(files),1).reshape(len(files),20)
print(rand_seeds)

missing_perc = 45
for i in range(1):#len(files)):
    file = files[i]
    # make the prediction
    start_time = time.time()
    X_test, Y_test, lon, lat, time_, meandata, mask = Test_Data_Loader(file, dim=(20,160,160), ntime_win=5, obs_err_std = [1.], 
                                                                       rand_seeds=rand_seeds[i,:], 
                                                                       miss=missing_perc,
                                                                       mean_olci = None,
                                                                       mean_multi = None)
    
    preds = np.zeros((X_test.shape[0], X_test.shape[1], X_test.shape[2], 2))
    ys = np.zeros((X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    
    for x in range(X_test.shape[0]):
        X_single = X_test[x,:,:,:]
        X_single = np.expand_dims(X_single, axis=0)
        pred = model.predict(X_single)
        preds[x,:,:,:] = pred[0,:,:,:]
        ys[x,:,:] = Y_test[x,:,:]
    
    with open(os.path.join(DATA_PATH_OUT,f'Prediction_times.txt'), 'a') as f:
        f.write(str(time.time() - start_time)+", ")
    
    # save the prediction
    savesample(os.path.join(DATA_PATH_OUT, "PRED_" + str(missing_perc) + "_" + os.path.basename(file) ), 
               preds[:,:,:,0], preds[:,:,:,1], lon, lat, time_, meandata, mask=mask)
    savesample(os.path.join(DATA_PATH_OUT, "X_" + str(missing_perc) + "_" + os.path.basename(file) ), 
                X_test[:,:,:,0], X_test[:,:,:,1], lon, lat, time_, meandata, x=True, mask=mask)
    savesample(os.path.join(DATA_PATH_OUT, "Y_" + str(missing_perc) + "_" + os.path.basename(file) ), 
            ys[:,:,:], ys[:,:,:], lon, lat, time_, meandata, y=True, mask=mask)
t = time.process_time() - start

# %%
