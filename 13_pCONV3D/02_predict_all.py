#%%
%matplotlib ipympl
import os
import sys
import numpy as np
import rasterio as rio
import time
import tensorflow as tf
from netCDF4 import Dataset, num2date
import h5py

from skimage import io as skio
from skimage import util as skutil

# Import modules from lib/ directory
from lib.STpconvUnet_chl import STpconvUnet
from lib.DataGenerator_chl_nc import DataGenerator, Test_Data_Loader_all_olci
import time
np.random.seed(42)
import tensorflow as tf
import datetime as dt
import shap
import xarray as xr

#%%
print("Using TensorFlow version", tf.__version__)

model_name = "model_213_KTT_dataperc30-100_3"
DATA_PATH_IN_TEMP = "../CHL_DATA/InputData_KTT/"

#%%
# if "save_weight_only" is set to True load model like this:
model = STpconvUnet.load(f"{model_name}_architecture", 
                         weights_name = f"{model_name}/epoch_77.h5")

#%%
data_olci = xr.open_dataset(os.path.join(DATA_PATH_IN_TEMP,f'input_data_olci.nc'))
print(data_olci)
start = time.process_time()

start_time = time.time()
Xs, masks, val_masks, landmasks, ys = Test_Data_Loader_all_olci(os.path.join(DATA_PATH_IN_TEMP,f'input_data_olci.nc'))

preds_ = np.zeros((0,Xs[0].shape[1], Xs[0].shape[2]))
ys_ = np.zeros((0, ys[0].shape[1], ys[0].shape[2]))
cloud_mask_ = np.zeros((0, masks[0].shape[1], masks[0].shape[2]))

for x in range(len(Xs)):
    X_single = Xs[x]
    mask = masks[x]
    Y_single = ys[x][0,:,:,:,0]
    cloud_mask = masks[x][0,:,:,:,0]
    landmask = landmasks[x][0,:,:,:,0]
    print(cloud_mask.shape)
    
    print("Landmaskshape")
    print(landmask.shape)
    
    print("maskshape")
    print(mask.shape)

    pred = model.predict([X_single, mask, mask, mask])

    pred = pred.swapaxes(2,3).swapaxes(1,2)

    Y_single = Y_single.swapaxes(1,2).swapaxes(0,1)
    landmask = landmask.swapaxes(1,2).swapaxes(0,1)
    cloud_mask = cloud_mask.swapaxes(1,2).swapaxes(0,1)

    pred = np.where(Y_single == 0, pred[0,:,:,:, 0], Y_single)
    pred = np.where(landmask==0, np.nan, pred)
    preds_ = np.append(preds_, pred, axis=0)
    cloud_mask_ = np.append(cloud_mask_, cloud_mask, axis=0)

#%% Convert data back to original values
chl_data = data_olci['CHL'].values
chl_min = np.nanmin(chl_data)
chl_max = np.nanmax(chl_data)
preds_trans = np.exp(preds_)
preds_trans = np.where(preds_trans>chl_max, chl_max, preds_trans)

print(preds_trans.shape)

#%%
data_olci

#%%
# save data
data_olci['CHL_interpolated_pCONV3D'] = (('time', 'lat', 'lon'), preds_trans)

#%%
data_olci['cloudmask'] = (('time', 'lat', 'lon'), cloud_mask_)

data_olci

#%%
data_olci.to_netcdf(os.path.join(DATA_PATH_IN_TEMP,f'input_data_olci_interpolated_pCONV3D_FINAL.nc'))
    
#%%
data_olci = xr.open_dataset(os.path.join(DATA_PATH_IN_TEMP,f'input_data_olci_interpolated_pCONV3D_FINAL.nc'))
print(data_olci)
data_olci.close()

import matplotlib.pyplot as plt  
plt.imshow(data_olci['CHL_interpolated_pCONV3D'][150,:,:], vmin=0, vmax=5)
plt.show()
#%%
import matplotlib.pyplot as plt   
for i in range(18,24):
    plt.figure()
    plt.imshow(preds_trans[i,:,:])
    plt.colorbar()
    plt.show()
    
np.nanmax(preds_trans)
#%%
plt.figure()
plt.imshow(preds_[30,:,:])
plt.show()



# %%
