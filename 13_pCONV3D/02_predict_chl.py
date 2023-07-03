#%%
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
from lib.DataGenerator_chl_nc import DataGenerator, Test_Data_Loader
import time
np.random.seed(42)

## uncomment for computation time measurements
# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)

#%%
print("Using TensorFlow version", tf.__version__)

DATA_PATH_IN = "../CHL_DATA/InputData_GTL/TestData_NC_GTL"

model_name = "model_213_GTL_dataperc30-100_4"

DATA_PATH_OUT = f"./training_results/prediction_{model_name}"

#%%
if os.path.exists(DATA_PATH_OUT):
    if len(os.listdir(DATA_PATH_OUT)) > 0:
        sys.exit("Output directory exists and is not empty")
else:
    os.makedirs(DATA_PATH_OUT)

# model = STpconvUnet.load("models/S5P_CO_model", weights_name = "models/S5P_CO_model.h5")
#%%
model = STpconvUnet.load(f"{model_name}_architecture", 
                         weights_name = f"{model_name}/epoch_80.h5") #73


def predict_single(img_path, out_path_pred, out_path_x, out_path_y, out_path_val,
                   out_path_mask, out_path_landmask, rand_seeds=None, missing_perc=None,
                   dim=(160,160,20)):
    X_temp, y = Test_Data_Loader(img_path, dim, rand_seeds, missing_perc)
    print(f"Min and max value of X: {np.min(X_temp[0])} and {np.max(X_temp[0])}")
    print(f"Min and max value of y: {np.min(y)} and {np.max(y)}")
    X, mask, val_mask, landmask = X_temp
    print(X.shape)
    pred = model.predict([X, mask, mask, mask]) # 2nd mask is not needed here, so we simply use the same mask
    print(pred.shape)
    print(f"Min and max value of pred: {np.min(pred)} and {np.max(pred)}")
    
    
    # create output file and copy spatial reference from input
    crs = None
    affine = None

    new_dataset_pred = rio.open(out_path_pred, 'w',driver='GTiff', height=pred.shape[1],
                           width=pred.shape[2],count=pred.shape[3], dtype=pred.dtype, crs=crs, transform=affine)
    new_dataset_x = rio.open(out_path_x, 'w',driver='GTiff', height=X.shape[1],
                           width=X.shape[2],count=X.shape[3], dtype=X.dtype, crs=crs, transform=affine)
    new_dataset_y = rio.open(out_path_y, 'w',driver='GTiff', height=y.shape[1],
                           width=y.shape[2],count=y.shape[3], dtype=y.dtype, crs=crs, transform=affine)
    new_dataset_val = rio.open(out_path_val, 'w',driver='GTiff', height=y.shape[1],
                           width=y.shape[2],count=y.shape[3], dtype=y.dtype, crs=crs, transform=affine)
    new_dataset_mask = rio.open(out_path_mask, 'w',driver='GTiff', height=y.shape[1],
                           width=y.shape[2],count=y.shape[3], dtype=y.dtype, crs=crs, transform=affine)
    new_dataset_landmask = rio.open(out_path_landmask, 'w',driver='GTiff', height=y.shape[1],
                           width=y.shape[2],count=y.shape[3], dtype=y.dtype, crs=crs, transform=affine)
    for it in range(model.nt):
        new_dataset_pred.write(np.where(landmask[0,:,:,it,0]!=0,pred[0,:,:,it,0],np.nan), it +1)
        new_dataset_x.write(np.where(X[0,:,:,it,0]!=0,X[0,:,:,it,0],np.nan), it +1)
        new_dataset_y.write(np.where(y[0,:,:,it,0]!=0,y[0,:,:,it,0],np.nan), it +1)
        new_dataset_val.write(val_mask[0,:,:,it,0], it +1)
        new_dataset_mask.write(mask[0,:,:,it,0], it +1)
        new_dataset_landmask.write(landmask[0,:,:,it,0], it +1)
    new_dataset_pred.close()
    new_dataset_x.close()
    new_dataset_y.close()

    
files = [
            os.path.join(DATA_PATH_IN, fname)
            for fname in sorted(os.listdir(DATA_PATH_IN))
            if fname.endswith(".nc") and fname.startswith("file_") 
            and "00-10" not in fname and "10-20" not in fname
            #and '50-100' in fname
        ]

print("MASK FIELS")
print(files)


start = time.process_time()
# generate the random seeds for the perlin noise generator, consistent for model comparison
rand_seeds = np.arange(0,20*len(files),1).reshape(len(files),20)
print(rand_seeds)
missing_perc = None
with open(os.path.join(DATA_PATH_OUT,f'Prediction_times.txt'), 'w') as f:
        f.write("Times: ")
for i in range(len(files)):
    start_time = time.time()
    out_path_pred = os.path.join(DATA_PATH_OUT, "PRED_"+str(missing_perc) +"_"+ os.path.basename(files[i]).split(".")[0] + '.tiff')
    out_path_x = os.path.join(DATA_PATH_OUT, "X_" +str(missing_perc) +"_" + os.path.basename(files[i]).split(".")[0] + '.tiff')
    out_path_y = os.path.join(DATA_PATH_OUT, "Y_"+str(missing_perc) +"_" + os.path.basename(files[i]).split(".")[0] + '.tiff')
    out_path_val = os.path.join(DATA_PATH_OUT, "VALMASK_"+str(missing_perc) +"_" + os.path.basename(files[i]).split(".")[0] + '.tiff')
    out_path_mask = os.path.join(DATA_PATH_OUT, "MASK_"+str(missing_perc) +"_" + os.path.basename(files[i]).split(".")[0] + '.tiff')
    out_path_landmask = os.path.join(DATA_PATH_OUT, "LANDMASK_"+str(missing_perc) +"_" + os.path.basename(files[i]).split(".")[0] + '.tiff')
    predict_single(files[i], out_path_pred, out_path_x, out_path_y, out_path_val, 
                   out_path_mask, out_path_landmask, rand_seeds[i], missing_perc)
    with open(os.path.join(DATA_PATH_OUT,str(missing_perc) +"_"+'Prediction_times.txt'), 'a') as f:
        f.write(str(time.time() - start_time)+", ")
t = time.process_time() - start  


print(t/len(files))
print("DONE.")


# %%
