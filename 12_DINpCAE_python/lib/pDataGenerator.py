#%%
#%matplotlib ipympl
import numpy as np
import tensorflow as tf
import os
import random
from netCDF4 import Dataset, num2date, MFDataset
import h5py
import matplotlib.pyplot as plt
from perlin_numpy import (
    generate_perlin_noise_2d, generate_fractal_noise_2d
)
import matplotlib.animation as animation
from tqdm import tqdm
from scipy import stats


def perlin_noise_generator(t,n, rand_seed=None, missing = None, train=True):
    '''
    Perlin Noise as taken from: 
    https://github.com/pvigier/perlin-numpy/blob/master/perlin_numpy/perlin2d.py
    Randomly assigning perlin noise to training images and defined percentage of missing data for testing
    
    Parameters
    ----------
    t : int
        number of images to generate
    n : int
        size of the image
    rand_seed : int
        random seed for reproducibility
    missing : int
        percentage of missing data
    train : bool
        if True, generate training data, else generate testing data
    '''
    binary_noise = np.zeros((t, n, n))
    for i in range(t):
        if rand_seed is not None:
            np.random.seed(rand_seed[i])
            
        # randomly allocate the 'missing' percentage of the area
        if train:
            missing = np.random.choice([45, 55, 65, 75, 85, 95], size=1,p=[0.1, 0.2, 0.2, 0.2, 0.2, 0.1])[0]    
            
        temp_noise = generate_fractal_noise_2d(shape=(n, n), 
                                            res=(2, 2), octaves=3,
                                            persistence=0.4,
                                            lacunarity=2)
        random_noise_dense = np.random.choice([0,1], size=(n,n),p=[0.5,0.5])
        random_noise = np.random.choice([0,1], size=(n,n),p=[0.9,0.1])
        if missing is None:
            binary_noise[i,:,:] = np.where(np.abs(temp_noise) < 0.35*np.max(np.abs(temp_noise)), 0, random_noise)
            binary_noise[i,:,:] = np.where(np.abs(temp_noise) < 0.46*np.max(np.abs(temp_noise)), binary_noise[i,:,:], random_noise_dense)
            binary_noise[i,:,:] = np.where(np.abs(temp_noise) < 0.58*np.max(np.abs(temp_noise)), binary_noise[i,:,:], 1)
        if missing == 45:
            binary_noise[i,:,:] = np.where(np.abs(temp_noise) < 0.65*np.max(np.abs(temp_noise)), 0, random_noise)
            binary_noise[i,:,:] = np.where(np.abs(temp_noise) < 0.75*np.max(np.abs(temp_noise)), binary_noise[i,:,:], random_noise_dense)
            binary_noise[i,:,:] = np.where(np.abs(temp_noise) < 0.85*np.max(np.abs(temp_noise)), binary_noise[i,:,:], 1)
        elif missing == 55:
            binary_noise[i,:,:] = np.where(np.abs(temp_noise) < 0.40*np.max(np.abs(temp_noise)), 0, random_noise)
            binary_noise[i,:,:] = np.where(np.abs(temp_noise) < 0.50*np.max(np.abs(temp_noise)), binary_noise[i,:,:], random_noise_dense)
            binary_noise[i,:,:] = np.where(np.abs(temp_noise) < 0.60*np.max(np.abs(temp_noise)), binary_noise[i,:,:], 1)
        elif missing == 65:
            binary_noise[i,:,:] = np.where(np.abs(temp_noise) < 0.25*np.max(np.abs(temp_noise)), 0, random_noise)
            binary_noise[i,:,:] = np.where(np.abs(temp_noise) < 0.36*np.max(np.abs(temp_noise)), binary_noise[i,:,:], random_noise_dense)
            binary_noise[i,:,:] = np.where(np.abs(temp_noise) < 0.48*np.max(np.abs(temp_noise)), binary_noise[i,:,:], 1)        
        elif missing == 75:
            binary_noise[i,:,:] = np.where(np.abs(temp_noise) < 0.10*np.max(np.abs(temp_noise)), 0, random_noise)
            binary_noise[i,:,:] = np.where(np.abs(temp_noise) < 0.20*np.max(np.abs(temp_noise)), binary_noise[i,:,:], random_noise_dense)
            binary_noise[i,:,:] = np.where(np.abs(temp_noise) < 0.30*np.max(np.abs(temp_noise)), binary_noise[i,:,:], 1)
        elif missing == 85:
            temp_noise = generate_fractal_noise_2d(shape=(n, n), 
                                            res=(1, 1), octaves=1,
                                            persistence=0.2,
                                            lacunarity=1)
            binary_noise[i,:,:] = np.where(np.abs(temp_noise) < 0.05*np.max(np.abs(temp_noise)), 0, random_noise)
            binary_noise[i,:,:] = np.where(np.abs(temp_noise) < 0.10*np.max(np.abs(temp_noise)), binary_noise[i,:,:], random_noise_dense)
            binary_noise[i,:,:] = np.where(np.abs(temp_noise) < 0.20*np.max(np.abs(temp_noise)), binary_noise[i,:,:], 1)
        elif missing == 95:
            temp_noise = generate_fractal_noise_2d(shape=(n, n), 
                                            res=(1, 1), octaves=1,
                                            persistence=0.2,
                                            lacunarity=1)
            binary_noise[i,:,:] = np.where(np.abs(temp_noise) < 0.02*np.max(np.abs(temp_noise)), 0, random_noise)
            binary_noise[i,:,:] = np.where(np.abs(temp_noise) < 0.05*np.max(np.abs(temp_noise)), binary_noise[i,:,:], random_noise_dense)
            binary_noise[i,:,:] = np.where(np.abs(temp_noise) < 0.08*np.max(np.abs(temp_noise)), binary_noise[i,:,:], 1)
        else:
            binary_noise[i,:,:] = np.where(np.abs(temp_noise) < 0.35*np.max(np.abs(temp_noise)), 0, random_noise)
            binary_noise[i,:,:] = np.where(np.abs(temp_noise) < 0.46*np.max(np.abs(temp_noise)), binary_noise[i,:,:], random_noise_dense)
            binary_noise[i,:,:] = np.where(np.abs(temp_noise) < 0.58*np.max(np.abs(temp_noise)), binary_noise[i,:,:], 1)
        
    
    
    return np.array(binary_noise, dtype=bool)

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, input_dir, batch_size=20, dim=(20,160,160), n_channels=1,
                 shuffle=True, fill = 0, varname="CHL", obs_err_std = [1.],
                 jitter_std = [0.05], ntime_win=3, cb = True, mean_olci = None, mean_multi = None):
        """
        Data generator for the training of the neural network.
        
        Parameters
        ----------
        input_dir : str
            Path to the directory containing the training data.
        batch_size : int
            Size of the batches.
        dim : tuple
            Dimensions of the input data.
        n_channels : int
            Number of channels of the input data.
        shuffle : bool
            Whether to shuffle the data at the end of each epoch.
        fill : float
            Value to fill the missing data with.
        varname : str
            Name of the variable to predict.
        obs_err_std : list
            List of observation error standard deviations.
        jitter_std : list
            List of jitter standard deviations.
        ntime_win : int
            Number of time steps in the input data.
        mean_olci : float
            Mean of the OLCI data.
        mean_multi : float
            Mean of the multi-sensor data.

        """
        self.dim = dim
        self.batch_size = batch_size
        self.input_dir = input_dir
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.fill = fill
        self.varname = varname
        self.obs_err_std = obs_err_std
        self.jitter_std = jitter_std
        self.ntime_win = ntime_win
        self.mean_olci = mean_olci
        self.mean_multi = mean_multi
        self.missing = perlin_noise_generator
        self.cb = cb # define whether to boxcox transform the data
        self.files = sorted(
            [
                os.path.join(input_dir, fname)
                for fname in os.listdir(input_dir)
                if fname.endswith(".nc") and fname.startswith("file_") 
                and not fname.endswith("X.nc") and not fname.endswith("Y.nc") 
                and not '00-10' in fname and not "10-20" in fname
                and not "20-30" in fname # and not "30-40" in fname
                #and not "40-50" in fname
                #and "olci" in fname
            ])

        self.n = len(self.files)

        self.file_infos = []
        self.load_gridded_nc()

        assert len(self.file_infos) == len(self.files)

        self.on_epoch_end()

    def __getitem__(self, index):

        X = np.empty((self.batch_size, self.dim[1], self.dim[2], 2 * self.ntime_win + 5))
        MASK = np.empty((self.batch_size, self.dim[1], self.dim[2], 1))
        y = np.empty((self.batch_size, self.dim[1], self.dim[2], 2))

        # print("file_index: ", index)
        # print("time_index: ", index)

        for i in range(self.batch_size):
            ii = index*self.batch_size + i
            it_prop = self.time_proportion[i]
            lon = self.file_infos[ii][0]
            lat = self.file_infos[ii][1]
            time = self.file_infos[ii][2]
            data =  self.file_infos[ii][3]
            data_mask = self.file_infos[ii][4]
            missing = self.missing(1,self.dim[1])
            mask = self.file_infos[ii][5] 
            ds_type = self.file_infos[ii][6]
            
            #print(self.files[ii])
            
            #mask = np.flipud(mask)

            # filter out data that is all nan as we do not want to train on such data
            not_all_nan = np.any(~np.isnan(data), axis=(1,2))
            indices = np.where(not_all_nan)[0]
            
            # set the time index to be among the non-nan indices
            it = int(np.round(it_prop * (len(indices)-1)))
            it = indices[it]
            
            # construct full x array to keep temporal information
            sz = data.shape
            ntime = sz[0]
            dayofyear = np.array([d.timetuple().tm_yday for d in time])
            dayofyear_cos = np.cos(2 * np.pi * dayofyear/365.25)
            dayofyear_sin = np.sin(2 * np.pi * dayofyear/365.25)
            
            if ds_type == "olci":
                if self.mean_olci is None:
                    meandata = np.nanmean(data, axis=0,keepdims=True)
                else:
                    meandata = self.mean_olci
            elif ds_type == "multi":
                if self.mean_multi is None:
                    meandata = np.nanmean(data, axis=0,keepdims=True)
                else:
                    meandata = self.mean_multi
            
            onemask = np.where(data == 0, 1, 0 )
            data = data - meandata
            data = np.where(data_mask == 1, 0, data)
            data = np.where(onemask == 1, 0, data)
            data = np.nan_to_num(data)


            x = np.zeros((sz[0],sz[1],sz[2],6),dtype="float32")
            x[:,:,:,0] = data  / (self.obs_err_std[0]**2)
            x[:,:,:,1] = 1-data_mask  / (self.obs_err_std[0]**2)


            lon_scaled = 2 * (lon - np.min(lon)) / (np.max(lon) - np.min(lon)) - 1
            lat_scaled = 2 * (lat - np.min(lat)) / (np.max(lat) - np.min(lat)) - 1

            x[:,:,:,2] = lon_scaled.reshape(1,1,len(lon))
            x[:,:,:,3] = lat_scaled.reshape(1,len(lat),1)
            x[:,:,:,4] = dayofyear_cos.reshape(len(dayofyear_cos),1,1)
            x[:,:,:,5] = dayofyear_sin.reshape(len(dayofyear_sin),1,1)

            nvar = 2 * self.ntime_win + 5

            xin = np.zeros((sz[1],sz[2],nvar),dtype="float32")
            xin[:,:,0:6]  = x[it,:,:,:]

            ioffset = 6
            for time_index in range(0,self.ntime_win):
                # nn is centered on the current time, e.g. -1 (past), 0 (present), 1 (future)
                nn = time_index - (self.ntime_win//2)
                # current time is already included, skip it
                if nn != 0:
                    i_clamped = min(ntime-1,max(0,it+nn))
                    xin[:,:,ioffset:(ioffset + 2)] = x[i_clamped,:,:,0:(2)]
                    ioffset = ioffset + 2

            # add missing data randomly based perlin noise
            selmask = missing[0,:,:]
            xin[:,:,0][selmask] = 0
            xin[:,:,1][selmask] = 0

            # add jitter
            xin[:,:,0] += self.jitter_std[0] * np.random.randn(sz[1],sz[2])
            xin[:,:,6] += self.jitter_std[0] * np.random.randn(sz[1],sz[2])
            xin[:,:,8] += self.jitter_std[0] * np.random.randn(sz[1],sz[2])
            
            # add a channel that indicates the type of data
            if ds_type == "multi":
                xin[:,:,-1] = np.zeros((sz[1],sz[2]))
            elif ds_type == "olci":
                xin[:,:,-1] = np.ones((sz[1],sz[2]))

            X[i,] = xin
            y[i,] = x[it,:,:,0:2]
            MASK[i,:,:,0] = np.where(np.equal(xin[it,:,0], 0), 0.0, 1.0)
            MASK[i,:,:,0] = np.where(np.equal(mask, 0), 1.0, MASK[i,:,:,0])

        return [X, MASK], y


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n/ self.batch_size))

    def on_epoch_end(self):
        if self.shuffle == True:
            order = random.sample(range(self.n),self.n)
            self.files  = [self.files [i] for i in order]
            self.file_infos = [self.file_infos[i] for i in order]
            self.time_proportion = np.random.random(size=self.dim[0])
            np.random.shuffle(self.time_proportion)

    def load_gridded_nc(self, minfrac = 0.05):
        for fname in self.files:
            lon,lat,time,data,missing,mask, ds_type = load_gridded_nc(fname,self.varname,minfrac)
            self.file_infos.append([lon,lat,time,data,missing,mask, ds_type])


def load_gridded_nc(fname, varname, minfrac = 0.05, cb=False):
    
    ds = Dataset(fname)
    lon = ds.variables["lon"][:].data
    lat = ds.variables["lat"][:].data
    time = num2date(ds.variables["time"][:],ds.variables["time"].units)

    data = ds.variables[varname][:,:,:]
    
    print(np.nanmax(data))
    print(np.nanmin(data))
    
    
    data = np.log(data)
    
    # multiply data by 10
    data = data*10

    if "mask" in ds.variables:
        #mask = ds.variables["mask"][0,:,:].data == 1
        mask = ds.variables["mask"][:,:].data == 1
    else:
        print("compute mask for ",varname,": sea point should have at least ",
            minfrac," for valid data tought time")

        if np.isscalar(data.mask):
            mask = np.ones((data.shape[1],data.shape[2]),dtype=np.bool)
        else:
            mask = np.mean(~data.mask,axis=0) > minfrac

    # generate the data mask
    ds.close()
    if np.isscalar(data.mask):
        missing = np.zeros(data.shape,dtype=np.bool)
    else:
        missing = data.mask.copy()

        
    #data[data.mask] = 0
    
    # transform data
    if 'multi' in fname:
        lambda_cb = -0.2816107171659248
        ds_type = 'multi'
        max_val = 194.84649658203125
        min_val = 0.12770438194274902
    elif 'olci' in fname:
        lambda_cb = -0.08346259836955258
        ds_type = 'olci'
        max_val = 171.9263916015625
        min_val = 1.000003457069397
    
    # if cb == True:
        # coxbox transformation
        #data = np.where(data == 0,0,(data**lambda_cb-1)/(lambda_cb))
    #data = (data**lambda_cb-1)/(lambda_cb)
        
        # # normalize
        #data = (data - min_val_cb)/(max_val_cb - min_val_cb)
        
    # else:
        # normalize
    #data = (data - min_val)/(max_val - min_val)
        # pass
    
    # multiply data by 10
    #data = data*10
    
    print(np.nanmax(data))
    print(np.nanmin(data))

    return lon,lat,time,data,missing,mask, ds_type

#%%
def Test_Data_Loader(file, dim=(2540,160,160), ntime_win=3, obs_err_std = [1.], cb=False, 
                     rand_seeds = None, miss = None, mean_olci = None, mean_multi= None):
    """
    Function to load Test data to test the trained model

    Parameters
    ----------
    file : str
        path to the file.
    dim : tuple, optional
        dimension of the data. The default is (20,160,160).
    ntime_win : int, optional
        number of time steps to include in the input. The default is 3.
    obs_err_std : list, optional
        list of observation error standard deviation. The default is [1.].
    rand_seeds : list, optional
        list of random seeds. The default is None.
    miss : list, optional
        list of missing data. The default is None.
    mean_olci : list, optional
        list of mean values for olci data. The default is None.
    mean_multi : list, optional
        list of mean values for multi data. The default is None.
        
    """

    file_infos = load_gridded_nc(file,"CHL" ,cb=cb)
    ntime = dim[0]

    X = np.empty((ntime, dim[1], dim[2], 2 * ntime_win + 5))
    Y = np.empty((ntime, dim[1], dim[2]))
    MASK = np.empty((ntime, dim[1], dim[2], 1))

    lon = file_infos[0]
    lat = file_infos[1]
    time = file_infos[2]
    data =  file_infos[3]
    missing = perlin_noise_generator(dim[0],dim[1], rand_seeds, miss)
    data_mask = file_infos[4]
    mask = file_infos[5]
    ds_type = file_infos[6]

    sz = data.shape
    ntime = sz[0]

    dayofyear = np.array([d.timetuple().tm_yday for d in time])
    dayofyear_cos = np.cos(2 * np.pi * dayofyear/365.25)
    dayofyear_sin = np.sin(2 * np.pi * dayofyear/365.25)    
    
    if ds_type == 'olci':
        if mean_olci is None:
            meandata = np.nanmean(data, axis=0,keepdims=True)
        else:
            meandata = mean_olci
    if ds_type == 'multi':
        if mean_multi is None:
            meandata = np.nanmean(data, axis=0,keepdims=True)
        else:
            meandata = mean_multi
        
    onemask = np.where(data == 0, 1, 0 )
    data = data - meandata
    data = np.where(data_mask == 1, 0, data)
    data = np.where(onemask == 1, 0, data)
    data = np.nan_to_num(data)

    x = np.zeros((sz[0],sz[1],sz[2],6),dtype="float32")

    x[:,:,:,0] = data
    x[:,:,:,0] = np.where(x[:,:,:,0]==-9999.0, 0 , x[:,:,:,0])
    x[:,:,:,1] = 1-data_mask

    lon_scaled = 2 * (lon - np.min(lon)) / (np.max(lon) - np.min(lon)) - 1
    lat_scaled = 2 * (lat - np.min(lat)) / (np.max(lat) - np.min(lat)) - 1

    x[:,:,:,2] = lon_scaled.reshape(1,1,len(lon))
    x[:,:,:,3] = lat_scaled.reshape(1,len(lat),1)
    x[:,:,:,4] = dayofyear_cos.reshape(len(dayofyear_cos),1,1)
    x[:,:,:,5] = dayofyear_sin.reshape(len(dayofyear_sin),1,1)

    nvar = 2 * ntime_win + 5
    
    print(data.shape)

    for t in range(ntime):
        xin = np.zeros((sz[1],sz[2],nvar),dtype="float32")
        xin[:,:,0:6]  = x[t,:,:,:]
        
        # add a channel that indicates the type of data
        if ds_type == "multi":
            xin[:,:,-1] = np.zeros((sz[1],sz[2]))
        elif ds_type == "olci":
            xin[:,:,-1] = np.ones((sz[1],sz[2]))

        ioffset = 6
        for time_index in range(0,ntime_win):
            # nn is centered on the current time, e.g. -1 (past), 0 (present), 1 (future)
            nn = time_index - (ntime_win//2)
            # current time is already included, skip it
            if nn != 0:
                i_clamped = min(ntime-1,max(0,t+nn))
                xin[:,:,ioffset:(ioffset + 2)] = x[i_clamped,:,:,0:2]
                ioffset = ioffset + 2

        
        yin = data[t].copy()
        selmask = missing[t,:,:]
        xin[:,:,0][selmask] = 0
        xin[:,:,1][selmask] = 0
               

        X[t,] = xin
        Y[t,] = yin
        MASK[t,:,:,0] = np.where(np.equal(xin[:,:,0], 0), 0.0, 1.0)
        MASK[t,:,:,0] = np.where(np.equal(mask[:,:], 0), 1.0, MASK[t,:,:,0])
        
    

    return [X, MASK], Y, lon, lat, time, meandata, mask
# %%
