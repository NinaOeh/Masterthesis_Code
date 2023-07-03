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

from netCDF4 import Dataset, num2date, MFDataset

# Import modules from lib/ directory
from lib.pDINCAEModel import DINpCAE
from lib.pDataGenerator import DataGenerator, Test_Data_Loader
from lib.pConv2D import pConv2D
import time
from keras.models import load_model, Model
import tensorflow as tf
import datetime as dt
import shap
import xarray as xr

np.random.seed(42)

## uncomment for computation time measurements
# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)

#%%
print("Using TensorFlow version", tf.__version__)

model_name = "pDINCAE_1_1_1_0_3_0_2_A_F_S_S_dataperc30-100_totalmean_mse"
DATA_PATH_IN = "../../CHL_DATA/InputData_RIGA/*/file_multi*.nc"
DATA_PATH_IN_TEMP = "../../CHL_DATA/InputData_RIGA/"
DATA_PATH_OUT = f"../../CHL_DATA/FilledData_RIGA"

#%%
if os.path.exists(DATA_PATH_OUT):
    if len(os.listdir(DATA_PATH_OUT)) > 0:
        sys.exit("Output directory exists and is not empty")
else:
    os.makedirs(DATA_PATH_OUT)

# model = STpconvUnet.load("models/S5P_CO_model", weights_name = "models/S5P_CO_model.h5")

#%%
# if "save_weight_only" is set to True load model like this:
model = DINpCAE.load(f"../data/models_pDINCAE/{model_name}_architecture", 
                    weights_name = f"../data/models_pDINCAE/{model_name}/epoch_470.h5")
    

#%%
def savesample(fname,m_rec,σ2_rec,lon,lat, time_, meandata, cloud_mask, real_data, mask, y=False, x=False):
    fill_value = -9999.
    
    # todo apply transfun to sigma_rec
    mask = np.where(mask == 0, 1, 0)
    sigma_rec = np.sqrt(np.abs(σ2_rec))
    
    if y == True:
        real_data = np.where(real_data == fill_value, np.nan, real_data) 
        real_data = np.where(real_data == 0, np.nan, real_data) 
        recdata = np.where(m_rec == fill_value, np.nan, m_rec) 
        recdata = np.where(recdata == 0, np.nan, recdata) 
    elif x == True:
        real_data = np.where(real_data == 0, np.nan, real_data) 
        recdata = np.where(m_rec == 0, np.nan, m_rec)
    else:
        recdata = m_rec
        
        mean_of_mean = np.nanmean(meandata)        
        meandata = np.where(np.isnan(meandata), mean_of_mean, meandata)

    
    recdata = recdata  + meandata
    real_data = real_data + meandata      
    
    print(meandata.shape)
    
    # Uncomment to only fill the gaps and keep the rest of the data as it is!!
    # recdata = np.where(cloud_mask[:,:,:,0]==0, recdata, real_data)
    

    # create file
    root_grp = Dataset(fname, 'w', format='NETCDF4')

    # dimensions
    root_grp.createDimension('time', None)
    root_grp.createDimension('lon', len(lon))
    root_grp.createDimension('lat', len(lat))

    # variables
    nc_lon = root_grp.createVariable('lon', 'f4', ('lon',))
    nc_lat = root_grp.createVariable('lat', 'f4', ('lat',))
    timeseries = root_grp.createVariable('time', float, ('time',))
    nc_meandata = root_grp.createVariable(
        'meandata', 'f4', ('lat','lon'),
        fill_value=fill_value)
    nc_landmask = root_grp.createVariable(
        'landmask', 'f4', ('lat', 'lon',),
        fill_value=fill_value)

    nc_mean_rec = root_grp.createVariable(
        'mean_rec', 'f4', ('time', 'lat', 'lon'),
        fill_value=fill_value)

    nc_sigma_rec = root_grp.createVariable(
        'sigma_rec', 'f4', ('time', 'lat', 'lon',),
        fill_value=fill_value)
    nc_cloudmask = root_grp.createVariable(
        'cloudmask', 'f4', ('time', 'lat', 'lon',),
        fill_value=fill_value)
    
    time_units = f"days since 1900-01-01 00:00:00"
    time_ = np.array([dt.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in time_])
    # kepp only the date without time
    time_ = time_.astype('datetime64[s]')
    time_= time_.astype(dt.datetime)

    mdate = np.array([(t_ - dt.datetime(1900, 1, 1)).total_seconds() / (24 * 60 * 60) for t_ in time_])

    timeseries.units = time_units
    timeseries[:] = mdate
    
    print(f"shape of cloudmask: {cloud_mask.shape}")

    # data
    nc_lon[:] = lon
    nc_lat[:] = lat
    nc_meandata[:,:] = meandata
    nc_landmask[:,:] = mask
    print(m_rec.shape)
    for n in range(m_rec.shape[0]):
        nc_mean_rec[n,:,:] = np.ma.masked_array(
            recdata[n,:,:],mask)
        nc_sigma_rec[n,:,:] = np.ma.masked_array(
            sigma_rec[n,:,:],mask)
        nc_cloudmask[n,:,:] = cloud_mask[n,:,:,0]

    root_grp.close()


#%%
    
files = DATA_PATH_IN

data = xr.open_mfdataset(files)
data.to_netcdf(os.path.join(DATA_PATH_IN_TEMP,f'input_data_multi.nc'))
#%%

missing_perc = 0
with open(os.path.join(DATA_PATH_OUT,f'Prediction_times.txt'), 'w') as f:
        f.write("Times: ")
start = time.process_time()

start_time = time.time()
X_test, Y_test, lon, lat, time_, meandata, mask = Test_Data_Loader(os.path.join(DATA_PATH_IN_TEMP,f'input_data_olci.nc'),
                                                                dim=(2540,160,160), 
                                                                ntime_win=5, obs_err_std = [1.],
                                                                cb = False, rand_seeds = None,
                                                                miss = missing_perc)

preds = np.zeros((X_test[0].shape[0], X_test[0].shape[1], X_test[0].shape[2], 2))
ys = np.zeros((Y_test.shape[0], Y_test.shape[1], Y_test.shape[2]))

for x in range(X_test[0].shape[0]):
    X_single = X_test[0][x,:,:,:]
    mask_single = X_test[1][x,:,:,:]
    Y_single = Y_test[x,:,:]
    X_single = np.expand_dims(X_single, axis=0)
    mask_single = np.expand_dims(mask_single, axis=0)
    pred = model.predict([X_single, mask_single])
    print(pred.shape)
    preds[x,:,:,:] = pred[0,:,:,:]
    ys[x,:,:] = Y_single[:,:]
    

with open(os.path.join(DATA_PATH_OUT,f'Prediction_times.txt'), 'a') as f:
    f.write(str(time.time() - start_time)+", ")
    
fig, ax = plt.subplots(1,3, figsize=(10,5))
ax[0].imshow(X_test[0][10,:,:,0])
ax[1].imshow(Y_test[10,:,:])
ax[2].imshow(preds[10,:,:,0])
plt.show()


# save the prediction
savesample(os.path.join(DATA_PATH_OUT, "AllFilled_DINpCAE.nc"), 
        preds[:,:,:,0], preds[:,:,:,1], lon, lat, time_, meandata, X_test[1], X_test[0][:,:,:,0], mask)

t = time.process_time() - start



# %%
