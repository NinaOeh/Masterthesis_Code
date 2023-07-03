#%%
# import all necessary libraries
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from scipy.interpolate import griddata, NearestNDInterpolator
import scipy
import getpass
import subprocess
import os
import random
import netCDF4
import h5netcdf
import rioxarray
import ffmpeg
import rasterio
import tifffile

from pykrige.ok3d import OrdinaryKriging3D
from pykrige.ok import OrdinaryKriging
from skimage import io as skio
from pykrige.uk3d import UniversalKriging3D

import gstools as gs
from sklearn.metrics import r2_score
import time

np.random.seed(123)

###-----------------------------------------------------------------------------------------------------------------###
###-----------------------------RUN ALL FUNCTIONS-------------------------------------------------------------------###
###-----------------------------------------------------------------------------------------------------------------###

def interpolate_nan_values_nearest(da):
    '''
        Function to interpolate the nan values in a DataArray
    '''
    da_temp = da
    # da_temp = da.swapaxes(1,2).swapaxes(0,1)
    # print(da_temp.shape)
    points = da_temp.ravel()
    valid = ~np.isnan(points)
    points_valid = points[valid]

    x = range(da_temp.shape[0])
    y = range(da_temp.shape[1])
    z = range(da_temp.shape[2])

    xx, yy, zz = np.meshgrid(x, y, z)
    xx, yy, zz = xx.ravel(), yy.ravel(), zz.ravel()

    print(xx.shape, yy.shape, zz.shape)

    xxv = xx[valid]
    yyv = yy[valid]
    zzv = zz[valid]

    print(xxv.shape, yyv.shape, zzv.shape)

    # feed these into the interpolator, and also provide the target grid
    interpolated = griddata(np.stack([xxv, yyv, zzv]).T, 
                            points_valid, (xx, yy, zz), 
                            method="nearest")

    # reshape to match the original array and replace the DataArray values with
    # the interpolated data
    da = interpolated.reshape(da_temp.shape)
    
    #da = da.swapaxes(0,1).swapaxes(1,2)
        
    return da


def interpolate_nan_values_linear(da):
    '''
        Function to interpolate the nan values in a DataArray
    '''
    # da_temp = da.swapaxes(0,1).swapaxes(1,2)
    # print(da_temp.shape)
    points = da.ravel()
    valid = ~np.isnan(points)
    points_valid = points[valid]

    x = range(da.shape[0])
    y = range(da.shape[1])
    z = range(da.shape[2])

    xx, yy, zz = np.meshgrid(x, y, z)
    xx, yy, zz = xx.ravel(), yy.ravel(), zz.ravel()

    print(xx.shape, yy.shape, zz.shape)

    xxv = xx[valid]
    yyv = yy[valid]
    zzv = zz[valid]

    print(xxv.shape, yyv.shape, zzv.shape)

    # feed these into the interpolator, and also provide the target grid
    interpolated = griddata(np.stack([xxv, yyv, zzv]).T, 
                            points_valid, (xx, yy, zz), 
                            method="linear")

    # reshape to match the original array and replace the DataArray values with
    # the interpolated data
    da = interpolated.reshape(da.shape)
        
    return da

def interpolate_nan_IterativeImputer(da):
    '''
        Function to interpolate the nan values in a DataArray
    '''
    # Create an instance of ItertiveImputer with sample_posterior=True
    imputer = IterativeImputer(sample_posterior=True)
    imputer.set_params(keep_empty_features=True)
    imputer.fit(da.reshape(da.shape[0], da.shape[1]*da.shape[2]))

    # Fit and transform the data to fill in missing values
    X_imputed = imputer.transform(da.reshape(da.shape[0], da.shape[1]*da.shape[2])).reshape(da.shape)

    return X_imputed

def interpolate_nan_KNNImputer(da):
    '''
        Function to interpolate the nan values in a DataArray
    '''
    da_temp = da.swapaxes(0,1).swapaxes(1,2)
    # Create an instance of KNNImputer with k=5
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    imputer.set_params(keep_empty_features=True)
    imputer.fit(da.reshape(da_temp.shape[0], da_temp.shape[1]*da_temp.shape[2]))

    # Fit and transform the data to fill in missing values
    X_imputed = imputer.transform(da_temp.reshape(da_temp.shape[0], da_temp.shape[1]*da_temp.shape[2])).reshape(da_temp.shape)
    
    print(X_imputed.shape)
    X_imputed = X_imputed.swapaxes(1,2).swapaxes(0,1)

    return X_imputed

def interpolate_nan_values_kriging3D(da):
    '''
        Function to interpolate the nan values in a DataArray
    '''
    # create arrays of the observed data and missing data locations
    observed_data = da[~np.isnan(da)]
    observed_locations = np.array(np.where(~np.isnan(da)))
    
    z = np.linspace(0,da.shape[0]-1,da.shape[0])
    x = np.linspace(0,da.shape[1]-1,da.shape[1])
    y = np.linspace(0,da.shape[2]-1,da.shape[2])
    
    values = observed_data.ravel()
    
    print("Let the kriging start")
    
    OK3d = OrdinaryKriging3D(
        observed_locations[0, :], 
        observed_locations[1, :], 
        observed_locations[2, :], 
        values, 
        variogram_model="linear"
    )
    k3d1, ss3d = OK3d.execute("grid", x, y, z)
    
    
def interpolate_nan_values_kriging2D(da):
    '''
        Function to interpolate the nan values in a DataArray
    '''
    def twoDkriging(data):
        # create arrays of the observed data and missing data locations
        if any(data[~np.isnan(data)]):
            try:
                observed_data = data[~np.isnan(data)]
                observed_locations = np.array(np.where(~np.isnan(data))).T
                
                x = np.linspace(0,data.shape[0]-1,data.shape[0])
                y = np.linspace(0,data.shape[1]-1,data.shape[1])
                
                values = observed_data.ravel()
                
                print("Let the kriging start")
                OK = OrdinaryKriging(
                    observed_locations[:, 1],
                    observed_locations[:, 0],
                    values,
                    variogram_model="linear",
                    verbose=False,
                    enable_plotting=False,
                )
                # interpolate data on an output grid
                z, ss = OK.execute("grid", x, y)
                return z
            except:
                print("Error in kriging")
                return data
        
        else:
            print("No data to interpolate")
            return data
    
    output = np.zeros_like(da)
    for x in range(da.shape[2]):
        output[:,:,x] = twoDkriging(da[:,:,x])
    
    return output    
    

def fill_nan_values_with_mean(da):
    nan_data = np.where(np.isnan(da)) #chl_np_array

    # get array of all indices with nan values
    non_x_indices = nan_data[0]
    non_y_indices = nan_data[1]
    non_time_indices = nan_data[2]

    # fill the values with the mean of the not-nan values
    da[non_x_indices, 
       non_y_indices, 
       non_time_indices] = np.nanmean(da)
    return da

def fill_nan_values_with_tempmean(da):
    print("here")       
    mean_array= np.expand_dims(np.nanmean(da, axis=2), axis=2)
    fill_array = np.repeat(mean_array, da.shape[2], axis=2)
    
    print(fill_array.shape)
    
    # fill the values with the mean of the not-nan values
    da = np.where(np.isnan(da), fill_array, da)
    return da


def r_mean_squared_error(y_true, y_pred, landmask=None):    
    
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    
    y_true_new = y_true[~np.isnan(y_pred)]
    y_pred_new = y_pred[~np.isnan(y_pred)]
    
    # subtract y_true from y_pred ignoring nan values
    subtracted = np.subtract(y_true_new, y_pred_new)
    rmse = np.sqrt(np.mean(subtracted**2))
    return rmse

def R_squared(y_true, y_pred):
    print("R squared")
    y_true_new_ = y_true[~np.isnan(y_pred)]
    y_true_new = y_true_new_[~np.isnan(y_true_new_)]
    
    y_pred_new_ = y_pred[~np.isnan(y_true)]
    y_pred_new = y_pred_new_[~np.isnan(y_pred_new_)]
    corr_matrix = np.corrcoef(y_true_new, y_pred_new)
    corr = corr_matrix[0,1]
    R_sq = corr**2
    return R_sq

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    print(n)
    m, se = np.nanmean(a), scipy.stats.sem(a, nan_policy='omit')
    h = se * scipy.stats.t.ppf((1+confidence)/2., n-1)
    return m, h

#%%


####---------------------------------------------------------------------------####
####-------------- RUN EVERYTHING FOR THE TEST DATASET-------------------------####
####---------------------------------------------------------------------------####

name = 'olci'
print("Start the test dataset interpolation")
#### Read the data
def interpolation(init_path, interpolation_type):
    print("Let's read the data")
    y_files_name = [
            os.path.join(init_path, fname)
            for fname in sorted(os.listdir(init_path))
            if fname.endswith(".tiff") and fname.startswith("Y_") 
            and name not in fname
        ]

    y_files = [
            np.exp(skio.imread(os.path.join(init_path, fname)))
            for fname in sorted(os.listdir(init_path))
            if fname.endswith(".tiff") and fname.startswith("Y_") 
            and name not in fname
        ]
    
    x_files = [
            np.exp(skio.imread(os.path.join(init_path, fname)))
            for fname in sorted(os.listdir(init_path))
            if fname.endswith(".tiff") and fname.startswith("X_") 
            and name not in fname
        ]

    valmask_files = [
            skio.imread(os.path.join(init_path, fname))
            for fname in sorted(os.listdir(init_path))
            if fname.endswith(".tiff") and fname.startswith("VALMASK_") 
            and name not in fname
        ]

    landmask_files = [
            skio.imread(os.path.join(init_path, fname))
            for fname in sorted(os.listdir(init_path))
            if fname.endswith(".tiff") and fname.startswith("LANDMASK_") 
            and name not in fname
        ]

    mask_files = [
            skio.imread(os.path.join(init_path, fname))
            for fname in sorted(os.listdir(init_path))
            if fname.endswith(".tiff") and fname.startswith("MASK_") 
            and name not in fname
        ]
        
    rmses = []
    r_sqrt = []
        
    print("The looping starts")
    print(len(y_files))
    for i in range(15,len(y_files)):
        # try:
            # convert the data to numpy arrays
            chl_np_array = np.array(y_files[i][:,:,:])
            random_val_mask = np.array(valmask_files[i])[:,:,:]
            land_mask = np.array(landmask_files[i][:,:,:])
            mask = np.array(mask_files[i][:,:,:])
            chl_np_val_masked = np.array(x_files[i][:,:,:])
            chl_np_val_masked = np.where(chl_np_val_masked == 0, np.nan, chl_np_val_masked)
            
            
            if interpolation_type == "linear":

                ### Linear 3D Interpolation
                print("Start the TriLinear Interpolation")
                chl_temp1 = chl_np_val_masked.copy()
                start_time = time.time()
                chl_interpolated = interpolate_nan_values_linear(chl_temp1)
                linear3D_time = time.time()-start_time
                print(f"Linear Interpolation took {time.time()-start_time} seconds.")
                
                
            if interpolation_type == "mean":

                ### Linear 3D Interpolation
                print("Start the Mean Interpolation")
                chl_temp1 = chl_np_val_masked.copy()
                
                start_time = time.time()
                chl_interpolated = fill_nan_values_with_mean(chl_temp1)
                mean3D_time = time.time()-start_time
                print(f"Interpolation took {time.time()-start_time} seconds.")
                
            if interpolation_type == "tempmean":

                ### Linear 3D Interpolation
                print("Start the Mean Interpolation")
                chl_temp1 = chl_np_val_masked.copy()
                
                start_time = time.time()
                chl_interpolated = fill_nan_values_with_tempmean(chl_temp1)
                mean3D_time = time.time()-start_time
                print(f"Interpolation took {time.time()-start_time} seconds.")
                
                
            if interpolation_type == "nearest_neighbor":
                
                if np.nansum(chl_np_val_masked) != 0.0:

                    ### Linear 3D Interpolation
                    print("Start the Interpolation")
                    chl_temp1 = chl_np_val_masked.copy()
                    start_time = time.time()
                    chl_interpolated = interpolate_nan_values_nearest(chl_temp1)
                    mean3D_time = time.time()-start_time
                    print(f"Interpolation took {time.time()-start_time} seconds.")
                    
                    
            if interpolation_type == "kriging":
                
                if np.nansum(chl_np_val_masked) != 0.0:

                    ### Kriging 2D Interpolation
                    print("Start the Interpolation")
                    chl_temp1 = chl_np_val_masked.copy()
                    start_time = time.time()
                    chl_interpolated = interpolate_nan_values_kriging2D(chl_temp1)
                    mean3D_time = time.time()-start_time
                    print(f"Interpolation took {time.time()-start_time} seconds.")
                    
                    
            if interpolation_type == "KNNImputer":    
                if np.nansum(chl_np_val_masked) != 0.0:

                    ### Kriging 2D Interpolation
                    print("Start the Interpolation")
                    chl_temp1 = chl_np_val_masked.copy()
                    start_time = time.time()
                    chl_interpolated = interpolate_nan_KNNImputer(chl_temp1)
                    mean3D_time = time.time()-start_time
                    print(f"Interpolation took {time.time()-start_time} seconds.")
                    
            val_mask_masked = np.where(land_mask==1, random_val_mask , 0)
            val_mask_masked = np.where(mask==1, val_mask_masked , 0)
            val_mask_masked = np.where(val_mask_masked!=0, val_mask_masked , 0)
            val_mask_indices = np.where(val_mask_masked==1)
            
            plt.figure()
            plt.imshow(chl_interpolated[:,:,0])
            plt.colorbar()
            plt.show()

            print("Check if any data is nan")
            print(any(np.isnan(chl_np_array[val_mask_indices])))
            if any(np.isnan(chl_interpolated[val_mask_indices])):
                all_gaps_filled = False
            else:
                all_gaps_filled = True

            print("Calculate the RMSEs")
            # Interpolated linear
            interpolated_rmse = r_mean_squared_error(chl_np_array[val_mask_indices],
                                                chl_interpolated[val_mask_indices])
            rmses.append(interpolated_rmse)
            print(f"Interpolated RMSE: {interpolated_rmse}")
            
            print("Calculate the R^2")
        
            r2 = R_squared(chl_np_array[val_mask_indices],
                           chl_interpolated[val_mask_indices])
            print(f"Interpolated R^2: {r2}")
            r_sqrt.append(r2)
        
        
            file_number = os.path.basename(y_files_name[i]).split("_")[1:]
            print(file_number)
            file_number = file_number[0].split(".")[0]
            print(file_number)
            
            chl_interpolated = np.where(land_mask==0, chl_interpolated , np.nan)
            
        # except:
        #     pass
        
        #tifffile.imwrite(os.path.join(interpol_path, f"INTER_{interpolation_type}_" + "_".join(file_number)+".tiff"), chl_interpolated)
        
    rmse, conf_int = mean_confidence_interval(rmses)
    
    # make the info file and path
    interpol_path = f'{init_path}/{interpolation_type}_interpolated'
    if not os.path.exists(interpol_path):
        os.mkdir(interpol_path)
        
    # with open(os.path.join(interpol_path,f'{interpolation_type}_infos.txt'), 'w') as f:
    #     f.write('Mean Interpolation, Time, RMSE, Rsquare \n')
    # with open(os.path.join(interpol_path,f'{interpolation_type}_infos.txt'), 'a') as f:
    #     f.write(f'{file_number}, {time.time()-start_time}, {rmse} / {conf_int}, {np.mean(r_sqrt)} \n')
        
        
    print(f"Interpolated RMSE and confidence interval: {rmse}, {conf_int}")
    print("Mean R^2: ", np.nanmean(r_sqrt))
    
    
init_path = "../pCONV3D/training_results/prediction_model_213_GTL_dataperc30-100_4"
interpolation(init_path, "kriging")
# options: linear, mean, nearest_neighbor, kriging


# %%
# Interpolate the values of the true data without gaps 
DATA_PATH_IN_TEMP = "../CHL_DATA/InputData_RIGA/"



# load the dataset
data_olci = xr.open_dataset(os.path.join(DATA_PATH_IN_TEMP,f'input_data_olci.nc'))
data_multi = xr.open_dataset(os.path.join(DATA_PATH_IN_TEMP,f'input_data_multi.nc'))

# interpolate the data
data_olci_interpolated_nn = interpolate_nan_values_nearest(data_olci['CHL'].values)

data_olci['CHL_interpolated_nn'] = (('time', 'lat', 'lon'), data_olci_interpolated_nn)
data_olci.to_netcdf(os.path.join(DATA_PATH_IN_TEMP,f'input_data_olci_interpolated.nc'))

# %%

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    print(n)
    m, se = np.nanmean(a), scipy.stats.sem(a, nan_policy='omit')
    h = se * scipy.stats.t.ppf((1+confidence)/2., n-1)
    return m, h

print(mean_confidence_interval([0.34445541458877926,0.2741192606090104,0.27966225663186645,0.4431330983984189,0.4089535472445517,0.4973463431738323,0.24753962122328904,0.22952375564627836,0.8718455855091606,0.5266714438819013,0.2330065310035373,0.3407509611794433,0.2506193123621935,1.1706999665923725,0.3354706162244225,0.24705202605462043,0.49869329190416467,0.25155430595252,0.36356213705796964,0.30725524019719896,0.21995566588262225]))

# %%
