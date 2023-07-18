import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import xarray as xr
import os
import io
import imageio
from IPython.display import Image, display
from ipywidgets import widgets, Layout, HBox
import os
from netCDF4 import Dataset, num2date
from datetime import datetime

#%%
def load_gridded_nc(fname, varname, minfrac = 0.05):

    ds = Dataset(fname)
    
    lon = ds.variables["lon"][:].data
    lat = ds.variables["lat"][:].data
    time = num2date(ds.variables["time"][:],ds.variables["time"].units)

    data_mean = ds.variables['mean_rec'][:,:,:]
    #data_std = ds.variables['sigma_rec'][:,:,:]
    
    # convert data to original scale
    #data_mean = ((data_mean/10)*(-0.1745758230162425) + 1.0 )**(1/(-0.1745758230162425))
    #data_std = ((data_std/10)*(-0.1745758230162425) + 1.0 )**(1/(-0.1745758230162425))
    

    landmask = ds.variables["landmask"][:,:].data == 1
    cloudmask  = ds.variables['cloudmask'][:,:,:]

    return lon, lat, time, data_mean, landmask, cloudmask


#%%
def Data_Loader(folder_chl, folder_features, indexes, dim=(20,160,160)):
    """
    Function to load Test data to test the trained model

    Parameters
    ----------
    folder_chl : str
        Path to the folder containing the CHL data.
    folder_features : str
        Path to the folder containing the feature data.
    indexes : list
        List of indexes of the files to load.
    dim : tuple, optional
        Dimensions of the data. The default is (20,160,160).
    """

    file_infos = load_gridded_nc(folder_chl, folder_features, indexes,"CHL" )

    lon = file_infos[0]
    lat = file_infos[1]
    time = file_infos[2]
    data_mean =  file_infos[3]
    landmask = file_infos[5]
    cloudmask = file_infos[6]

    dayofyear = np.array([d.timetuple().tm_yday for d in time])
    dayofyear_cos = np.cos(2 * np.pi * dayofyear/365.25)
    dayofyear_sin = np.sin(2 * np.pi * dayofyear/365.25)

    x = np.zeros((dim[0],dim[1],dim[2],1),dtype="float32") #change last dimension to number of channels

    x[:,:,:,0] = data_mean
    print(data_mean)

    return x, landmask

def Data_Loader_Features(DATA_PATH_CHL, DATA_PATH_Features, 
                         interpolated_value, dim=(20,160,160), datatrans='log', feat='all'):
    """
    Function to load Test data to test the trained model, including all features

    Parameters
    ----------
    DATA_PATH_CHL : str
        Path to the folder containing the CHL data.
    DATA_PATH_Features : str
        Path to the folder containing the feature data.
    interpolated_value : str
        Name of the interpolated value to use.
    dim : tuple, optional
        Dimensions of the data. The default is (20,160,160).
    datatrans : str, optional
        Type of data transformation to use. The default is 'log'.
    feat : str, optional
        Features to use. The default is 'all'.
    """
    
    ##### Load the data #####

    data = xr.open_dataset(DATA_PATH_CHL)

    # extract the data
    chl_data_init = data['CHL'].values
    chl_data = data[interpolated_value].values
    landmask = data['mask'].values[0,:,:]
    chl_data = np.where(landmask == 0, np.nan, chl_data)
    cloudmask = np.where((np.isnan(chl_data_init)) & (landmask == 1), 1, 0)
    lon = data['lon'].values
    lat = data['lat'].values

    # get the max number of pixels that are not non in dimension 1 and 2
    max_pixels = np.sum(landmask)
    print(f"Max Pixels: {max_pixels}")

    # read all features 
    wind_v10 = np.load(f'{DATA_PATH_Features}/wind_v10.npy')
    wind_u10 = np.load(f'{DATA_PATH_Features}/wind_u10.npy')
    pres_t2m = np.load(f'{DATA_PATH_Features}/pres_t2m.npy')
    pres_tp = np.load(f'{DATA_PATH_Features}/pres_tp.npy')
    pres_rad = np.load(f'{DATA_PATH_Features}/pres_rad.npy')
    sst = np.load(f'{DATA_PATH_Features}/sst_values.npy')
    wave_VHM0 = np.load(f'{DATA_PATH_Features}/VHM0_data.npy')
    #wave_VHM0 = np.append(wave_VHM0 , np.load(f'{DATA_PATH_Features}/VHM02_data.npy'), axis=0)
    wave_VMDR = np.load(f'{DATA_PATH_Features}/VMDR_data.npy')
    #wave_VMDR = np.append(wave_VMDR , np.load(f'{DATA_PATH_Features}/VMDR2_data.npy'), axis=0)
    topo = np.load(f'{DATA_PATH_Features}/topo.npy')
    topo =  np.repeat(topo[np.newaxis, :, :], 2540, axis=0)
    
    # min-max normalize all features
    wind_v10 = (wind_v10 - np.nanmin(wind_v10))/(np.nanmax(wind_v10) - np.nanmin(wind_v10))
    wind_u10 = (wind_u10 - np.nanmin(wind_u10))/(np.nanmax(wind_u10) - np.nanmin(wind_u10))
    pres_t2m = (pres_t2m - np.nanmin(pres_t2m))/(np.nanmax(pres_t2m) - np.nanmin(pres_t2m))
    pres_tp = (pres_tp - np.nanmin(pres_tp))/(np.nanmax(pres_tp) - np.nanmin(pres_tp))
    pres_rad = (pres_rad - np.nanmin(pres_rad))/(np.nanmax(pres_rad) - np.nanmin(pres_rad))
    sst = (sst - np.nanmin(sst))/(np.nanmax(sst) - np.nanmin(sst))
    wave_VHM0 = (wave_VHM0 - np.nanmin(wave_VHM0))/(np.nanmax(wave_VHM0) - np.nanmin(wave_VHM0))
    wave_VMDR = (wave_VMDR - np.nanmin(wave_VMDR))/(np.nanmax(wave_VMDR) - np.nanmin(wave_VMDR))
    topo = (topo - np.nanmin(topo))/(np.nanmax(topo) - np.nanmin(topo))
    
    lon_scaled = (lon - np.min(lon)) / (np.max(lon) - np.min(lon))
    lat_scaled = (lat - np.min(lat)) / (np.max(lat) - np.min(lat))

    lon_scaled = lon_scaled.reshape(1,1,len(lon))
    lat_scaled = lat_scaled.reshape(1,len(lat),1)
    
    if datatrans == 'log':
        chl_min = np.nanmin(chl_data)
        chl_max = np.nanmax(chl_data)
        # log transform chl value
        chl_data = np.log(chl_data)
    elif datatrans == 'minmax':
        # min-max normalize chl data
        chl_min = np.nanmin(chl_data)
        chl_max = np.nanmax(chl_data)
        chl_data = (chl_data - chl_min)/(chl_max - chl_min)
    elif datatrans == "log_minmax":
        # log transform chl value
        chl_data = np.log(chl_data)
        # min-max normalize chl data
        chl_min = np.nanmin(chl_data)
        chl_max = np.nanmax(chl_data)
        chl_data = (chl_data - chl_min)/(chl_max - chl_min)
    
    # split all temporal data into 20 day blocks
    splitting_array = np.arange(20, chl_data.shape[0], 20)
    chl_np_array = np.split(chl_data, splitting_array, axis=0)
    chl_np_array_init = np.split(chl_data_init, splitting_array, axis=0)
    cloudmask_array = np.split(cloudmask, splitting_array, axis=0)
    wind_v10_array = np.split(wind_v10, splitting_array, axis=0)
    wind_u10_array = np.split(wind_u10, splitting_array, axis=0)
    pres_t2m_array = np.split(pres_t2m, splitting_array, axis=0)
    pres_tp_array = np.split(pres_tp, splitting_array, axis=0)
    pres_rad_array = np.split(pres_rad, splitting_array, axis=0)
    sst_array = np.split(sst, splitting_array, axis=0)
    wave_VHM0_array = np.split(wave_VHM0, splitting_array, axis=0)
    wave_VMDR_array = np.split(wave_VMDR, splitting_array, axis=0)
    topo_array = np.split(topo, splitting_array, axis=0)

    time_array = pd.date_range(start='2016-05-01', end='2023-04-14', freq='D')
    time_split = np.split(time_array, splitting_array, axis=0)


    datetime_objects = np.array([datetime.utcfromtimestamp(dt.astype('O') / 1e9) for dt in data.time.data])
    dayofyear = np.array([dt.timetuple().tm_yday for dt in datetime_objects])

    dayofyear_cos = np.cos(2 * np.pi * dayofyear/365.25)
    dayofyear_sin = np.sin(2 * np.pi * dayofyear/365.25)
    
    # normalize day of year
    dayofyear_cos = (dayofyear_cos - np.nanmin(dayofyear_cos))/(np.nanmax(dayofyear_cos) - np.nanmin(dayofyear_cos))
    dayofyear_sin = (dayofyear_sin - np.nanmin(dayofyear_sin))/(np.nanmax(dayofyear_sin) - np.nanmin(dayofyear_sin))
    
    dayofyear_cos_split = np.split(dayofyear_cos, splitting_array, axis=0)
    dayofyear_sin_split = np.split(dayofyear_sin, splitting_array, axis=0)
    
    if feat == 'all':

        x = np.zeros((0, dim[0],dim[1],dim[2], 15 ),dtype="float32") #change last dimension to number of channels


        for i in range(len(splitting_array)):
            if np.sum(~np.isnan(chl_np_array_init[i])) >= 0.3*20*max_pixels:
                x_temp = np.zeros((1, dim[0],dim[1],dim[2], 15 ),dtype="float32")
                x_temp[0, :,:,:,0] = chl_np_array[i]
                x_temp[0, :,:,:,1] = cloudmask_array[i]
                x_temp[0, :,:,:,2] = wind_v10_array[i]
                x_temp[0, :,:,:,3] = wind_u10_array[i]
                x_temp[0, :,:,:,4] = pres_t2m_array[i]
                x_temp[0, :,:,:,5] = pres_tp_array[i]
                x_temp[0, :,:,:,6] = pres_rad_array[i]
                x_temp[0, :,:,:,7] = sst_array[i]
                x_temp[0, :,:,:,8] = wave_VHM0_array[i]
                x_temp[0, :,:,:,9] = wave_VMDR_array[i]
                x_temp[0, :,:,:,10] = topo_array[i]
                x_temp[0, :,:,:,11] = dayofyear_cos_split[i].reshape(len(dayofyear_cos_split[i]),1,1)
                x_temp[0, :,:,:,12] = dayofyear_sin_split[i].reshape(len(dayofyear_sin_split[i]),1,1)
                x_temp[0, :,:,:,13] = lon_scaled
                x_temp[0, :,:,:,14] = lat_scaled
                
                x = np.append(x, x_temp, axis=0)
    else:
        x = np.zeros((0, dim[0],dim[1],dim[2], 1 ),dtype="float32") #change last dimension to number of channels


        for i in range(len(splitting_array)):
            if np.sum(~np.isnan(chl_np_array_init[i])) >= 0.3*20*max_pixels:
                x_temp = np.zeros((1, dim[0],dim[1],dim[2], 1 ),dtype="float32")
                x_temp[0, :,:,:,0] = chl_np_array_init[i]
                
                x = np.append(x, x_temp, axis=0)
        

    return x, landmask, chl_min, chl_max



def Data_Loader_Features_Pred(DATA_PATH_CHL, DATA_PATH_Features, 
                         interpolated_value, dim=(20,160,160)):
    """
    Function to load Test data to test the trained model

    Args:
        dim (tuple, optional): _description_. Defaults to (20,160,160).
        ntime_win (int, optional): _description_. Defaults to 3.
    """
    
    ##### Load the data #####
    x = np.zeros((0, 10,dim[1],dim[2], 15 ),dtype="float32") #change last dimension to number of channels
    files = [f for f in os.listdir(DATA_PATH_CHL) if f.endswith('.nc')]
    print(len(files))
    for i in range(len(files)):
        file = os.path.join(DATA_PATH_CHL, files[i])
        print(file)

        data = xr.open_dataset(file, decode_times=False)

        # extract the data
        chl_data_init = data['CHL'].values[10:]
        chl_data = data[interpolated_value].values[10:]
        landmask = data['mask'].values
        chl_data = np.where(landmask == 0, np.nan, chl_data)
        cloudmask = np.where((np.isnan(chl_data_init)) & (landmask == 1), 1, 0)
        lon = data['lon'].values
        lat = data['lat'].values

        # get the max number of pixels that are not non in dimension 1 and 2
        max_pixels = np.sum(landmask)
        print(f"Max Pixels: {max_pixels}")

        # read all features 
        wind_v10 = np.load(f'{DATA_PATH_Features}/wind_v10.npy')[i,10:,:,:]
        wind_u10 = np.load(f'{DATA_PATH_Features}/wind_u10.npy')[i,10:,:,:]
        pres_t2m = np.load(f'{DATA_PATH_Features}/t2m.npy')[i,10:,:,:]
        pres_tp = np.load(f'{DATA_PATH_Features}/tp.npy')[i,10:,:,:]
        pres_rad = np.load(f'{DATA_PATH_Features}/cdir.npy')[i,10:,:,:]
        sst = np.load(f'{DATA_PATH_Features}/SST.npy')[i,10:,:,:]
        wave_VHM0 = np.load(f'{DATA_PATH_Features}/VHM0.npy')[i,10:,:,:]
        wave_VMDR = np.load(f'{DATA_PATH_Features}/VMDR.npy')[i,10:,:,:]
        topo = np.load(f'{DATA_PATH_Features}/topo.npy')
        topo =  np.repeat(topo[np.newaxis, :, :], 10, axis=0)
        
        # min-max normalize all features
        wind_v10 = (wind_v10 - np.nanmin(wind_v10))/(np.nanmax(wind_v10) - np.nanmin(wind_v10))
        wind_u10 = (wind_u10 - np.nanmin(wind_u10))/(np.nanmax(wind_u10) - np.nanmin(wind_u10))
        pres_t2m = (pres_t2m - np.nanmin(pres_t2m))/(np.nanmax(pres_t2m) - np.nanmin(pres_t2m))
        pres_tp = (pres_tp - np.nanmin(pres_tp))/(np.nanmax(pres_tp) - np.nanmin(pres_tp))
        pres_rad = (pres_rad - np.nanmin(pres_rad))/(np.nanmax(pres_rad) - np.nanmin(pres_rad))
        sst = (sst - np.nanmin(sst))/(np.nanmax(sst) - np.nanmin(sst))
        wave_VHM0 = (wave_VHM0 - np.nanmin(wave_VHM0))/(np.nanmax(wave_VHM0) - np.nanmin(wave_VHM0))
        wave_VMDR = (wave_VMDR - np.nanmin(wave_VMDR))/(np.nanmax(wave_VMDR) - np.nanmin(wave_VMDR))
        topo = (topo - np.nanmin(topo))/(np.nanmax(topo) - np.nanmin(topo))
        
        lon_scaled = (lon - np.min(lon)) / (np.max(lon) - np.min(lon))
        lat_scaled = (lat - np.min(lat)) / (np.max(lat) - np.min(lat))

        lon_scaled = lon_scaled.reshape(1,1,len(lon))
        lat_scaled = lat_scaled.reshape(1,len(lat),1)
        
        chl_min = np.nanmin(chl_data)
        chl_max = np.nanmax(chl_data)
        # log transform chl value
        chl_data = np.log(chl_data)
        
        if i ==0:
            time_array = pd.date_range(start='2023-06-10', end='2023-06-19', freq='D')
        elif i ==1:
            time_array = pd.date_range(start='2023-06-11', end='2023-06-20', freq='D')
        elif i ==2:
            time_array = pd.date_range(start='2023-06-12', end='2023-06-21', freq='D')
        elif i ==3:
            time_array = pd.date_range(start='2023-06-13', end='2023-06-22', freq='D')
        elif i ==4:
            time_array = pd.date_range(start='2023-06-14', end='2023-06-23', freq='D')

        #print(pd.to_datetime(time_array))
        #datetime_objects = np.array([datetime.utcfromtimestamp(dt.astype('O') / 1e9) for dt in time_array])
        dayofyear = np.array([dt.timetuple().tm_yday for dt in time_array])

        dayofyear_cos = np.cos(2 * np.pi * dayofyear/365.25)
        dayofyear_sin = np.sin(2 * np.pi * dayofyear/365.25)
        
        # normalize day of year
        dayofyear_cos = (dayofyear_cos - np.nanmin(dayofyear_cos))/(np.nanmax(dayofyear_cos) - np.nanmin(dayofyear_cos))
        dayofyear_sin = (dayofyear_sin - np.nanmin(dayofyear_sin))/(np.nanmax(dayofyear_sin) - np.nanmin(dayofyear_sin))


        x_temp = np.zeros((1, dim[0],dim[1],dim[2], 15 ),dtype="float32")
        x_temp[0, :,:,:,0] = chl_data
        x_temp[0, :,:,:,1] = cloudmask
        x_temp[0, :,:,:,2] = wind_v10
        x_temp[0, :,:,:,3] = wind_u10
        x_temp[0, :,:,:,4] = pres_t2m
        x_temp[0, :,:,:,5] = pres_tp
        x_temp[0, :,:,:,6] = pres_rad
        x_temp[0, :,:,:,7] = sst
        x_temp[0, :,:,:,8] = wave_VHM0
        x_temp[0, :,:,:,9] = wave_VMDR
        x_temp[0, :,:,:,10] = topo
        x_temp[0, :,:,:,11] = dayofyear_cos.reshape(len(dayofyear_cos),1,1)
        x_temp[0, :,:,:,12] = dayofyear_sin.reshape(len(dayofyear_sin),1,1)
        x_temp[0, :,:,:,13] = lon_scaled
        x_temp[0, :,:,:,14] = lat_scaled
                
        x = np.append(x, x_temp, axis=0)        

    return x, landmask, chl_min, chl_max