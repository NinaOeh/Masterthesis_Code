# This file implements tf.keras.utils.Sequence to read
# spatiotemporal input blocks batch-wise when needed. This 
# is used while training models in order to reduce memory consumption.
#
# Please notice that negative values of pixels are interpreted as nodata values and will be 
# internally set to 0. For datasets with correct negative values, the loading procedure below
# must be adapted using a different nodata value. 
#
#%%

import numpy as np
import tensorflow as tf
import os
import random
import numpy as np
from netCDF4 import Dataset, num2date
import h5py
import matplotlib.pyplot as plt
from perlin_numpy import (
    generate_perlin_noise_2d, generate_fractal_noise_2d
)
import matplotlib.animation as animation
from tqdm import tqdm
from scipy import stats
from skimage import io as skio


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
            print(f"Random seed: {rand_seed[i]}")
            
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
    def __init__(self, input_dir, batch_size=32, dim=(160,160,20), n_channels=1, shuffle=True, nmax = -1, fill = 0, train = True):
        '''
        Data generator for the training of the neural network.
        
        Parameters
        ----------
        input_dir : str
            Path to the directory containing the input files.
        batch_size : int
            Size of the batches.
        dim : tuple
            Dimensions of the input data.
        n_channels : int
            Number of channels of the input data.
        shuffle : bool
            Whether to shuffle the data at the end of each epoch.
        nmax : int
            Maximum number of files to use.
        fill : int
            Value to fill the missing data with.
        train : bool
            Whether to use the training or validation data.
        '''
        self.dim = dim
        self.batch_size = batch_size
        self.input_dir = input_dir
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.fill = fill
        self.train = train
        
        self.files = sorted(
            [
                os.path.join(input_dir, fname)
                for fname in os.listdir(input_dir)
                if fname.endswith(".nc") and fname.startswith("file_") 
                and '00-10' not in fname and "10-20" not in fname 
                and '20-30' not in fname #and '30-40' not in fname 
                #and '40-50' not in fname
                #and 'olci' in fname
            ]
        )
        
        if nmax > 0 and nmax < len(self.files):
            order = random.sample(range(len(self.files)),nmax)
            self.files  = [self.files [i] for i in order]
        
        
        self.n = len(self.files)
        self.on_epoch_end()
        #self.max_chl = np.max[np.max(skio.imread(i)) for i in self.y_files]

    def __len__(self):
        return int(np.floor(self.n / self.batch_size))

    def __getitem__(self, index):
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.dim[2], 2))
        y = np.empty((self.batch_size, self.dim[0], self.dim[1], self.dim[2], 1))
        mask = np.empty((self.batch_size, self.dim[0], self.dim[1], self.dim[2], 1))
        val_mask = np.empty((self.batch_size, self.dim[0], self.dim[1], self.dim[2], 1))
        landmask = np.empty((self.batch_size, self.dim[0], self.dim[1], self.dim[2], 1))
       
        for i in range(self.batch_size):
            ii = index*self.batch_size + i
            
            # read the data
            file = self.files[ii]
            ds = Dataset(file)

            data = ds.variables['CHL'][:,:,:]
            land_mask = ds.variables['mask'][:,:]
            land_mask = np.tile(land_mask,(data.shape[0],1,1))
            
            # generate the fake clouds for validation
            validation_mask = perlin_noise_generator(data.shape[0],data.shape[1], train=self.train)
            
            # get the cloud mask from the data
            cloud_mask = data.mask.copy()
            
                
            
            
            if 'multi' in file:
                ds_type = np.zeros_like(data)
                lambda_cb = -0.2816107171659248
                max_val = 194.84649658203125
                min_val = 0.12770438194274902
            elif 'olci' in file:
                ds_type = np.ones_like(data)
                lambda_cb = -0.08346259836955258
                max_val = 171.9263916015625
                min_val = 1.000003457069397
                
            # transform data
            data_init = data.copy()
            
            # log transform the data
            data = np.log(data)
            
            # coxbox
            #data = (data**lambda_cb-1)/(lambda_cb)
                
            # normalize
            #data = (data - min_val)/(max_val - min_val)
            
            data[data_init.mask] = 0.0  

            # get a cloud mask: 0 is cloud, 1 is known data or land
            cloud_mask = np.where(cloud_mask==0, 1, 0)
            cloud_mask = np.where(land_mask==1, cloud_mask , 1)
        
            # omit data for training validation
            x_data = np.where(validation_mask == 0, data, 0)
            
            #x_data = np.expand_dims(x_data, axis=3)
            y_data = np.expand_dims(data, axis=3)
            cloud_mask = np.expand_dims(cloud_mask, axis=3)
            validation_mask = np.expand_dims(validation_mask, axis=3)
            land_mask = np.expand_dims(land_mask, axis=3)
            
                
            x_data = x_data.swapaxes(0,1).swapaxes(1,2)
            y_data = y_data.swapaxes(0,1).swapaxes(1,2)
            ds_type = ds_type.swapaxes(0,1).swapaxes(1,2)
            cloud_mask = cloud_mask.swapaxes(0,1).swapaxes(1,2)
            validation_mask = validation_mask.swapaxes(0,1).swapaxes(1,2)
            land_mask = land_mask.swapaxes(0,1).swapaxes(1,2)
            
            X[i,:,:,:,0] = x_data
            X[i,:,:,:,1] = ds_type
            y[i,:,:,:,:] = y_data
            mask[i,:,:,:,:] = cloud_mask
            val_mask[i,:,:,:,:] = validation_mask
            landmask[i,:,:,:,:] = land_mask
            
        return [X, mask, val_mask, landmask], y 

    def on_epoch_end(self):
        if self.shuffle == True:
            order = random.sample(range(self.n),self.n)
            self.files  = [self.files [i] for i in order]

#%%
def Test_Data_Loader(file, dim, rand_seeds=None, missing_perc=None, train=False):
    
        """
        Function to load Test data to test the trained model

        Parameters
        ----------
        file : str
            path to the file.
        dim : tuple, optional
            dimension of the data. The default is (20,160,160).
        rand_seeds : list, optional
            list of random seeds. The default is None.
        missing_perc : list, optional
            list of missing data. The default is None.
        train : bool, optional
            if True, the data is used for training. The default is False.
        """
    
        X = np.empty((1, dim[0], dim[1], dim[2], 2))
        y = np.empty((1, dim[0], dim[1], dim[2], 1))
        mask = np.empty((1, dim[0], dim[1], dim[2], 1))
        val_mask = np.empty((1, dim[0], dim[1], dim[2], 1))
        landmask = np.empty((1, dim[0], dim[1], dim[2], 1))
       
        
        # read the data
        ds = Dataset(file)

        data = ds.variables['CHL'][:,:,:]
        land_mask = ds.variables['mask'][:,:]
        land_mask = np.tile(land_mask,(data.shape[0],1,1))
        
        # generate the fake clouds for validation
        validation_mask = perlin_noise_generator(data.shape[0],data.shape[1], rand_seeds, missing_perc, train=train)
        
        validation_mask = np.zeros((data.shape[0], data.shape[1], data.shape[1]))
        # get the cloud mask from the data
        cloud_mask = data.mask.copy()
         
        
        if 'multi' in file:
                ds_type = np.zeros_like(data)
                lambda_cb = -0.2816107171659248
                max_val = 194.84649658203125
                min_val = 0.12770438194274902
        elif 'olci' in file:
            ds_type = np.ones_like(data)
            lambda_cb = -0.08346259836955258
            max_val = 171.9263916015625
            min_val = 1.000003457069397
            
        # transform data
        data_init = data.copy()    
        
        # log transform the data
        data = np.log(data)
        
        # coxbox
        #data = (data**lambda_cb-1)/(lambda_cb)
        
        # normalize
        #data = (data - min_val)/(max_val - min_val)     
        
        data[data_init.mask] = 0.0   
        

        # get a cloud mask: 0 is cloud, 1 is known data or land
        cloud_mask = np.where(cloud_mask==0, 1, 0)
        cloud_mask = np.where(land_mask==1, cloud_mask , 1)
    
        # omit data for training validation
        x_data = np.where(validation_mask == 0, data, 0)
        
        #x_data = np.expand_dims(x_data, axis=3)
        y_data = np.expand_dims(data, axis=3)
        cloud_mask = np.expand_dims(cloud_mask, axis=3)
        validation_mask = np.expand_dims(validation_mask, axis=3)
        land_mask = np.expand_dims(land_mask, axis=3)
        
            
        x_data = x_data.swapaxes(0,1).swapaxes(1,2)
        y_data = y_data.swapaxes(0,1).swapaxes(1,2)
        ds_type = ds_type.swapaxes(0,1).swapaxes(1,2)
        cloud_mask = cloud_mask.swapaxes(0,1).swapaxes(1,2)
        validation_mask = validation_mask.swapaxes(0,1).swapaxes(1,2)
        land_mask = land_mask.swapaxes(0,1).swapaxes(1,2)
        
        X[0,:,:,:,0] = x_data
        X[0,:,:,:,1] = ds_type
        y[0,:,:,:,:] = y_data
        mask[0,:,:,:,:] = cloud_mask
        val_mask[0,:,:,:,:] = validation_mask
        landmask[0,:,:,:,:] = land_mask        
            
        return [X, mask, val_mask, landmask], y 
# %%

def Test_Data_Loader_all_olci(file, dim=(20,160,160), rand_seeds=None):
    """
    Data Loader to fill entire dataset
    
    Parameters
    ----------
    file : str
        path to the file.
    dim : tuple, optional
        dimension of the data. The default is (20,160,160).
    rand_seeds : list, optional
        list of random seeds. The default is None.
    """
    # read the data
    ds = Dataset(file)
    Xs = []
    ys = []
    masks = []
    val_masks = []
    landmasks = []
    
    for i in range(0,ds.variables['CHL'].shape[0], 20):
        X = np.empty((1, dim[2], dim[1], dim[0], 2))
        y = np.empty((1, dim[2], dim[1], dim[0], 1))
        mask = np.empty((1, dim[2], dim[1], dim[0], 1))
        val_mask = np.empty((1, dim[2], dim[1], dim[0], 1))
        landmask = np.empty((1, dim[2], dim[1], dim[0], 1))

        data = ds.variables['CHL'][i:i+20,:,:]
        print(data.shape)
        land_mask = ds.variables['mask'][i:i+20,:,:]
        print(land_mask.shape)
        
        # no fake clouds added for final data filling
        validation_mask = np.zeros((dim[0], dim[1], dim[1]))
        
        # get the cloud mask from the data
        cloud_mask = data.mask.copy()
         
        ds_type = np.ones_like(data)
        lambda_cb = -0.08346259836955258
        max_val = 171.9263916015625
        min_val = 1.000003457069397
            
        # transform data
        data_init = data.copy()    
        
        # log transform the data
        data = np.log(data)
        
        # coxbox
        # data = (data**lambda_cb-1)/(lambda_cb)
        
        # normalize
        #data = (data - min_val)/(max_val - min_val)     
        
        data[data_init.mask] = 0.0   
        

        # get a cloud mask: 0 is cloud, 1 is known data or land
        cloud_mask = np.where(cloud_mask==0, 1, 0)
        cloud_mask = np.where(land_mask==1, cloud_mask , 1)
    
        # omit data for training validation
        x_data = np.where(validation_mask == 0, data, 0)
        
        #x_data = np.expand_dims(x_data, axis=3)
        y_data = np.expand_dims(data, axis=3)
        cloud_mask = np.expand_dims(cloud_mask, axis=3)
        validation_mask = np.expand_dims(validation_mask, axis=3)
        land_mask = np.expand_dims(land_mask, axis=3)
        
            
        x_data = x_data.swapaxes(0,1).swapaxes(1,2)
        y_data = y_data.swapaxes(0,1).swapaxes(1,2)
        ds_type = ds_type.swapaxes(0,1).swapaxes(1,2)
        cloud_mask = cloud_mask.swapaxes(0,1).swapaxes(1,2)
        validation_mask = validation_mask.swapaxes(0,1).swapaxes(1,2)
        land_mask = land_mask.swapaxes(0,1).swapaxes(1,2)
        
        X[0,:,:,:,0] = x_data
        X[0,:,:,:,1] = ds_type
        y[0,:,:,:,:] = y_data
        mask[0,:,:,:,:] = cloud_mask
        val_mask[0,:,:,:,:] = validation_mask
        landmask[0,:,:,:,:] = land_mask
        
        Xs.append(X)
        ys.append(y)
        masks.append(mask)
        val_masks.append(val_mask)
        landmasks.append(landmask)       
            
    return Xs, masks, val_masks, landmasks, ys
