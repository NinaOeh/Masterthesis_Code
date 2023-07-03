#%%
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import io
import imageio
from IPython.display import Image, display
from ipywidgets import widgets, Layout, HBox
import os
from netCDF4 import Dataset, num2date
from lib.DataLoader import Data_Loader, Data_Loader_Features
from sklearn.metrics import r2_score
import tqdm
import pandas as pd
import scipy
import seaborn as sns
from matplotlib.colors import ListedColormap

#%%
np.random.seed(42)


# Prepare CHL predicted data to be passed to LSTM model
DATA_PATH_CHL = "../CHL_DATA/InputData_RIGA/input_data_olci_interpolated_pCONV3D_FINAL.nc"
DATA_PATH_CHL_INIT = "../CHL_DATA/InputData_RIGA/input_data_olci.nc"
DATA_PATH_Features = f"../CHL_DATA/InputData_RIGA/Features"
MODELNAME = "model_3_2_1_1_1_1_RIGA"
from sklearn.model_selection import train_test_split

# load the model
model = keras.models.load_model(f"{MODELNAME}.h5")

# load the data
dim=(20,160,160)
#interpolated_value = 'CHL_interpolated_nn'
interpolated_value = 'CHL_interpolated_pCONV3D'

dataset, landmask, chl_min, chl_max = Data_Loader_Features(DATA_PATH_CHL, 
                                               DATA_PATH_Features, 
                                               interpolated_value, 
                                               dim=dim, 
                                               datatrans='log', feat='all')

# In the first step we will split the data in training and remaining dataset
X_train_ind, X_rem_ind, y_train, y_rem = train_test_split(np.arange(dataset.shape[0]),np.arange(dataset.shape[0]), 
                                                          train_size=0.8, random_state=42)

# Now since we want the valid and test size to be equal (10% each of overall data). 
# we have to define valid_size=0.5 (that is 50% of remaining data)
X_valid_ind, X_test_ind, y_valid, y_test = train_test_split(X_rem_ind,y_rem, test_size=0.5, 
                                                            random_state=42)
    
test_dataset = dataset[X_test_ind]

print(f"Test dataset shape: {test_dataset.shape}")
print(f"Test dataset min: {np.nanmin(test_dataset[:,:,:,:,0])}")
print(f"Test dataset max: {np.nanmax(test_dataset[:,:,:,:,0])}")


# fill the missing values with 0
test_dataset = np.nan_to_num(test_dataset)

#%%
# dates of the test dataset
splitting_array = np.arange(20, 2540, 20)
time_array = pd.date_range(start='2016-05-01', end='2023-04-14', freq='D')
time_split = np.split(time_array, splitting_array, axis=0)

for i in X_test_ind:
    print(i)
    print(time_split[i])
# Generate prediction for all files in the validation dataset and validate them
#%%
## Single step prediction
rmses = np.empty((test_dataset.shape[0], 10))
r2s = np.empty((test_dataset.shape[0], 10))
all_pred_frames = np.empty((test_dataset.shape[0], 20, 160, 160))
all_init_frames = np.empty((test_dataset.shape[0], 20, 160, 160))
# for i in range(test_dataset.shape[0]):
#     print(i)
#     example = test_dataset[i]
    
    
#     frames_input = example[:10, ...]
    
#     temp = np.zeros_like(example)
#     temp[:10,:,:,0] = example[:10,:,:,0].copy()
#     temp[:,:,:,1:] = example[:,:,:,1:].copy()
    
#     frames = example[:10,:,:,0]
#     original_frames = example[10:,:,:,0]
    
#     print(original_frames.shape)
    
#     for j in range(10):
#         # Extract the model's prediction and post-process it.
#         frames_input = temp[:10+j+1, ...]
#         new_prediction = model.predict(np.expand_dims(frames_input, axis=0))
#         new_prediction = np.squeeze(new_prediction, axis=0)
#         predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)
        
#         # Extend the set of prediction frames.
#         frames = np.concatenate((frames, predicted_frame[:,:,:,0]), axis=0)
        
#         temp[10+j,:,:,0] = predicted_frame[0,:,:,0]
        
#         # transform to original scale
#         predicted_frame_orig = predicted_frame * (chl_max - chl_min) + chl_min
#         predicted_frame_orig = np.exp(predicted_frame_orig)
        
#         original_frame = original_frames[j] * (chl_max - chl_min) + chl_min
#         original_frame = np.exp(original_frame)
        
#         print(np.max(predicted_frame_orig))
#         print(np.min(predicted_frame_orig))
#         print(np.max(original_frame))
#         print(np.min(original_frame))
    
#         # calculate the rmse and r2 score for the prediction per day
#         rmses[i,j] = np.sqrt(np.mean((original_frame - predicted_frame_orig)**2))
#         r2s[i,j] = r2_score(original_frame.flatten(), predicted_frame_orig.flatten())
        
#     all_pred_frames[i] = frames
#     all_init_frames[i] = example[:20,:,:,0]
    
    
## Single step prediction with features for 10 days
for i in range(test_dataset.shape[0]):
    print(i)
    example_init = test_dataset[i].copy()
    
    example = test_dataset[i].copy()
    
    frames_input = example[:10, ...]
    
    frames = example[:10,:,:,0]
    original_frames = example[10:,:,:,0]
    
    print(original_frames.shape)
    
    for j in range(10):
        # Extract the model's prediction and post-process it.
        frames_input = example[:10+j+1, ...]
        new_prediction = model.predict(np.expand_dims(frames_input, axis=0))
        print(new_prediction.shape)
        new_prediction = np.squeeze(new_prediction, axis=0)
        predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)
        
        # transform to original scale
        #predicted_frame = predicted_frame * (chl_max - chl_min) + chl_min
        predicted_frame = np.exp(predicted_frame)
        
        #original_frame = original_frames[j] * (chl_max - chl_min) + chl_min
        original_frame = np.exp(original_frame)
    
        # calculate the rmse and r2 score for the prediction per day
        rmses[i,j] = np.sqrt(np.mean((original_frame - predicted_frame_orig)**2))
        r2s[i,j] = r2_score(original_frame.flatten(), predicted_frame_orig.flatten())
        
        # Extend the set of prediction frames.
        frames = np.concatenate((frames, predicted_frame[:,:,:,0]), axis=0)
        example[10+j,:,:,0] = predicted_frame[0,:,:,0]
        
        print(np.max(predicted_frame))
        print(np.min(predicted_frame))
        print(np.max(original_frame))
        print(np.min(original_frame))
        
    all_pred_frames[i] = frames
    all_init_frames[i] = example_init[:,:,:,0]
 
 
#%%
## Multi step prediction   
history_size = 10
target_size = 5

def R_squared(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0,1]
    R_sq = corr**2
    return R_sq

# Create overlapping windows of history and target data.      
def create_overlapping_frames(dataset, history_size=10,
                      target_size=5):
    sz = dataset.shape
    X = np.empty((0,history_size+target_size,sz[2],sz[3], sz[4]))
    y = np.empty((0,history_size+target_size,sz[2],sz[3],1))
    start_index = 0
    start_index = start_index + history_size + target_size
    end_index = dataset.shape[1] + 1

    for i in range(start_index, end_index):
        d_temp = dataset[:,i-history_size-target_size:i,:,:,0].shape
        X_temp = np.empty((d_temp[0], d_temp[1], d_temp[2], d_temp[3], sz[4]))
        print(X_temp.shape)
        X_temp[:,:,:,:,:] = dataset[:,i-history_size-target_size:i,:,:,:]
        # X_temp[:,:,:,:,4] = dataset[:,i-history_size-target_size:i,:,:,5]
        # X_temp[:,:,:,:,3:8] = dataset[:,i-history_size-target_size:i,:,:,4:9]
        # X_temp[:,:,:,:,8:] = dataset[:,i-history_size-target_size:i,:,:,10:]
        X = np.append(X, X_temp, axis=0)
        y_temp = dataset[:,i-history_size-target_size:i,:,:,0]
        y_temp = np.expand_dims(y_temp, axis=-1)
        y = np.append(y, y_temp, axis=0)

    return X, y

test_dataset = create_overlapping_frames(test_dataset, history_size=history_size,
                                         target_size=target_size)
print(test_dataset[0].shape)

rmses = np.empty((test_dataset[0].shape[0], target_size))
r2s = np.empty((test_dataset[0].shape[0], target_size))
mae = np.empty((test_dataset[0].shape[0], target_size))
all_pred_frames = np.empty((test_dataset[0].shape[0], 15, 160, 160))
all_init_frames = np.empty((test_dataset[0].shape[0], 15, 160, 160))
for i in range(test_dataset[0].shape[0]):
    print(i)
    frames_input = test_dataset[0][i]
    frames_input = frames_input[:history_size, ...]
    
    example =test_dataset[1][i]
    
    frames = example[:history_size,:,:,0]
    original_frame = example[history_size:,:,:,0]
    
    print(original_frame.shape)
    
    print("Now on to the predictions")
    new_prediction = model.predict(np.expand_dims(frames_input, axis=0))
    
    # case parallel output
    # predicted_frame = [np.squeeze(n, axis=0) for n in new_prediction]
    # predicted_frame = [np.expand_dims(n[-1, ...], axis=0) for n in predicted_frame]
    # predicted_frame = [np.swapaxes(n, 0, 3) for n in predicted_frame]
    # predicted_frame = np.concatenate(predicted_frame, axis=0)
    # print(predicted_frame.shape)
    
    # case single multi output
    predicted_frame = np.squeeze(new_prediction, axis=0)
    predicted_frame = np.expand_dims(predicted_frame[-1, ...], axis=0)
    predicted_frame = np.swapaxes(predicted_frame, 0, 3)
    print(predicted_frame.shape)
    
        
    # Extend the set of prediction frames.
    frames = np.concatenate((frames, predicted_frame[:,:,:,0]), axis=0)
    
    # transform to original scale
    #predicted_frame = predicted_frame * (chl_max - chl_min) + chl_min
    predicted_frame = np.exp(predicted_frame)
    
    
    #original_frame = original_frame * (chl_max - chl_min) + chl_min
    original_frame = np.exp(original_frame)
    
    for j in range(target_size):        
        # calculate the rmse and r2 score for the prediction per day
        rmses[i,j] = np.sqrt(np.mean((original_frame[j,:,:] - predicted_frame[j,:,:,0])**2))
        r2s[i,j] = R_squared(np.nan_to_num(original_frame[j,:,:].flatten()), np.nan_to_num(predicted_frame[j,:,:,0]).flatten())
        mae[i,j] = np.mean(np.abs(original_frame[j,:,:] - predicted_frame[j,:,:,0]))
        
    all_pred_frames[i] = frames
    all_init_frames[i] = example[:,:,:,0]
    
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    print(n)
    m, se = np.nanmean(a), scipy.stats.sem(a, nan_policy='omit')
    h = se * scipy.stats.t.ppf((1+confidence)/2., n-1)
    return m, h

        
for i in range(target_size):
    print(f"RMSE for day {i+1}: {mean_confidence_interval(rmses[:,i])}")
    print(f"R2 for day {i+1}: {np.mean(r2s[:,i])}")
    print(f"MAE for day {i+1}: {np.mean(mae[:,i])}")

print(rmses.flatten())
print(mean_confidence_interval(rmses.flatten()))
print(np.nanmean(r2s))   

#%%
# Baseline analysis, using the last known day for the next 5 days
rmses_base = np.empty((test_dataset[0].shape[0], target_size))
r2s_base = np.empty((test_dataset[0].shape[0], target_size))
mae_base = np.empty((test_dataset[0].shape[0], target_size))
all_pred_frames_base = np.empty((test_dataset[0].shape[0], 15, 160, 160))
all_init_frames_base = np.exp(test_dataset[0][:,:,:,:,0])
for i in range(test_dataset[0].shape[0]):
    all_pred_frames_base[i,:history_size,:,:] = all_init_frames_base[i,:history_size,:,:]
    all_pred_frames_base[i,:history_size+1,:,:] = all_init_frames_base[i,history_size-1,:,:]
    all_pred_frames_base[i,:history_size+2,:,:] = all_init_frames_base[i,history_size-1,:,:]
    all_pred_frames_base[i,:history_size+3,:,:] = all_init_frames_base[i,history_size-1,:,:]
    all_pred_frames_base[i,:history_size+4,:,:] = all_init_frames_base[i,history_size-1,:,:]
    all_pred_frames_base[i,:history_size+5,:,:] = all_init_frames_base[i,history_size-1,:,:]
    for j in range(target_size):        
        # calculate the rmse and r2 score for the prediction per day
        rmses_base[i,j] = np.sqrt(np.mean((all_init_frames_base[i,j+10,:,:] - all_pred_frames_base[i,j+10,:,:])**2))
        r2s_base[i,j] = R_squared(np.nan_to_num(all_init_frames_base[i,j+10,:,:].flatten()), np.nan_to_num(all_pred_frames_base[i,j+10,:,:]).flatten())
        mae_base[i,j] = np.mean(np.abs(all_init_frames_base[i,j+10,:,:] - all_pred_frames_base[i,j+10,:,:]))
        
for i in range(target_size):
    print(f"Baseline RMSE for day {i+1}: {mean_confidence_interval(rmses_base[:,i])}")
    print(f"Baseline R2 for day {i+1}: {np.mean(r2s_base[:,i])}")
    print(f"Baseline MAE for day {i+1}: {np.mean(mae_base[:,i])}")

print(rmses_base.flatten())
print(mean_confidence_interval(rmses_base.flatten()))
print(np.nanmean(r2s_base))   

#%%
# Baseline 2 , average value of known 10 days
# Baseline analysis, using the last known day for the next 5 days
rmses_base = np.empty((test_dataset[0].shape[0], target_size))
r2s_base = np.empty((test_dataset[0].shape[0], target_size))
mae_base = np.empty((test_dataset[0].shape[0], target_size))
all_pred_frames_base = np.empty((test_dataset[0].shape[0], 15, 160, 160))
all_init_frames_base = np.exp(test_dataset[0][:,:,:,:,0])
for i in range(test_dataset[0].shape[0]):
    all_pred_frames_base[i,:history_size,:,:] = all_init_frames_base[i,:history_size,:,:]
    all_pred_frames_base[i,:history_size+1,:,:] = np.mean(all_init_frames_base[i,:history_size,:,:], axis=0)
    all_pred_frames_base[i,:history_size+2,:,:] = np.mean(all_init_frames_base[i,:history_size,:,:], axis=0)
    all_pred_frames_base[i,:history_size+3,:,:] = np.mean(all_init_frames_base[i,:history_size,:,:], axis=0)
    all_pred_frames_base[i,:history_size+4,:,:] = np.mean(all_init_frames_base[i,:history_size,:,:], axis=0)
    all_pred_frames_base[i,:history_size+5,:,:] = np.mean(all_init_frames_base[i,:history_size,:,:], axis=0)
    for j in range(target_size):        
        # calculate the rmse and r2 score for the prediction per day
        rmses_base[i,j] = np.sqrt(np.mean((all_init_frames_base[i,j+10,:,:] - all_pred_frames_base[i,j+10,:,:])**2))
        r2s_base[i,j] = R_squared(np.nan_to_num(all_init_frames_base[i,j+10,:,:].flatten()), np.nan_to_num(all_pred_frames_base[i,j+10,:,:]).flatten())
        mae_base[i,j] = np.mean(np.abs(all_init_frames_base[i,j+10,:,:] - all_pred_frames_base[i,j+10,:,:]))
        
for i in range(target_size):
    print(f"Baseline RMSE for day {i+1}: {mean_confidence_interval(rmses_base[:,i])}")
    print(f"Baseline R2 for day {i+1}: {np.mean(r2s_base[:,i])}")
    print(f"Baseline MAE for day {i+1}: {np.mean(mae_base[:,i])}")

print(rmses_base.flatten())
print(mean_confidence_interval(rmses_base.flatten()))
print(np.nanmean(r2s_base))   

#%%
# Select a random example from the validation dataset.
num = 21

frames = all_pred_frames[num]
original_frames = all_init_frames[num]
frames = np.where(landmask == 0, np.nan, frames)
original_frames = np.where(landmask == 0, np.nan, original_frames)

# frames = frames * (chl_max - chl_min) + chl_min
# original_frames = original_frames * (chl_max - chl_min) + chl_min

frames = np.exp(frames)
original_frames = np.exp(original_frames)

print(chl_min)
print(chl_max)

print(np.nanmax(frames))
print(np.nanmin(frames))

print(np.nanmax(original_frames))
print(np.nanmin(original_frames))

print(original_frames.shape)

# Construct a figure for the original and new frames.
fig, axes = plt.subplots(3, target_size+2, figsize=(18, 6))
history_size = 10

# Plot the original frames.
for idx, ax in enumerate(axes[0]):
    idx_ = idx + history_size-2
    im1 = ax.imshow(np.squeeze(original_frames[idx_,:,:]), vmin=0, vmax=40, origin='lower')
    
    if idx_ < 10:
        ax.set_title(f"Known day {idx_+1}")
        ax.tick_params(labelbottom=False, labelleft=False, length=0)
    else:
        ax.set_title(f"Predicted day {idx-1}")
        ax.tick_params(labelbottom=False, labelleft=False, length=0)

# Plot the new frames.
for idx, ax in enumerate(axes[1]):
    idx = idx + history_size-2
    im2 = ax.imshow(np.squeeze(frames[idx,:,:]), vmin=0, vmax=40, origin='lower')
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    
# Plot the Error
for idx, ax in enumerate(axes[2]):
    idx = idx + history_size-2
    im3 = ax.imshow(frames[idx,:,:]-original_frames[idx,:,:], vmin=-5, vmax=5, origin='lower', cmap='RdYlGn')
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    
axes[0, 0].set_ylabel("Original", fontsize=13)
axes[1, 0].set_ylabel("Predicted", fontsize=13)
axes[2, 0].set_ylabel("Error", fontsize=13)

# Display the figure.
plt.colorbar(im1, ax=axes[0, :].tolist(), pad=0.01)
plt.colorbar(im2, ax=axes[1, :].tolist(), pad=0.01)
plt.colorbar(im3, ax=axes[2, :].tolist(), pad=0.01)
plt.show()

#%% Show the historgram and error

def generate_pred_histogram(chl, preds, prefix, path, d, fig_height=4):
        sns.set_theme(style="whitegrid")
        print("Start generating histogram...")
        min_val = -1
        max_val = 6
        data = {"Original chl-a concentration [mg/L]": chl, "Predicted chl-a concentration [mg/L]": preds}

        g = sns.jointplot(
            x="Original chl-a concentration [mg/L]",
            y="Predicted chl-a concentration [mg/L]",
            kind="hist", # "kde"
            data=data,
            fill=True,
            #color="g",
            cmap="turbo",
            height=fig_height,
            xlim=(min_val, max_val),
            ylim=(min_val, max_val),
            cbar=True,
            cbar_kws={"label": "density"} # "format": formatter
        )
        # https://stackoverflow.com/questions/60845764/colorbar-for-sns-jointplot-kde-style-on-the-side
        plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
        pos_joint_ax = g.ax_joint.get_position()
        pos_marg_x_ax = g.ax_marg_x.get_position()
        g.ax_joint.set_position(
            [pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height]
        )
        g.fig.axes[-1].set_position([0.85, pos_joint_ax.y0, 0.07, pos_joint_ax.height])

        g.ax_joint.plot([min_val, max_val], [min_val, max_val], linestyle="--")
        if d != 'a':
            plt.suptitle(f"Day {d}")
        plt.savefig(
            os.path.join(path, f"{prefix}_histogram.png"), bbox_inches="tight", pad_inches=0.01
        )
        plt.show()
        sns.reset_defaults()
        
def generate_residuals_figure(ys_real, preds_real, prefix, path, i):
    print(i)
    # Define colormap
    cm = plt.cm.get_cmap('viridis')
    my_cmap = cm(np.linspace(0,1,10))

    # Create new colormap
    my_cmap = ListedColormap(my_cmap)
    fig, axes = plt.subplots(2, 1, figsize=(4, 6))
    axes[0].set_ylabel("Density", fontsize=13) 
    axes[1].set_ylabel("Residuals", fontsize=13)

    axes[0].grid()
    axes[1].grid()
    print(np.nanmax(ys_real))
    print(np.nanmax(preds_real))
    

    sns.kdeplot(data=ys_real, ax=axes[0], label="True", color=my_cmap(1), fill=True)
    sns.kdeplot(data=preds_real, ax=axes[0], label="Predicted", color='green', fill=True)
    axes[0].set_xlabel("Chl-a concentration \n [mg/m$^3$]", fontsize=13)
    axes[0].legend(fontsize=11)

    axes[1].scatter(ys_real, np.subtract(preds_real, ys_real), alpha=0.2, 
                color=my_cmap(4))
    axes[1].set_xlabel("Observed", fontsize=13)
    # if i == 1:
    #     axes[0].set_xlim(-5, 117)
    #     axes[1].set_xlim(-5, 117)
    #     axes[1].set_ylim(-120, 130)
    # elif i == 2:
    #     axes[0].set_xlim(-5, 113)
    #     axes[1].set_xlim(-5, 113)
    #     axes[1].set_ylim(-112, 160)
    # elif i == 3:
    #     axes[0].set_xlim(-5, 113)
    #     axes[1].set_xlim(-5, 113)
    #     axes[1].set_ylim(-112, 105)
    # elif i == 4:
    #     axes[0].set_xlim(-5, 113)
    #     axes[1].set_xlim(-5, 113)
    #     axes[1].set_ylim(-112, 130)
    # elif i == 5:
    #     axes[0].set_xlim(-5, 116)
    #     axes[1].set_xlim(-5, 116)
    #     axes[1].set_ylim(-112, 100)
        
    if i != 'a':
        fig.suptitle(f"Day {i}", fontsize=15)
    # elif i == 'a':
    #     axes[0].set_xlim(-5, 116)
    #     axes[1].set_xlim(-5, 116)
    #     axes[1].set_ylim(-120, 160)
    plt.tight_layout()
    
    
    plt.savefig(
            os.path.join(path, f"{prefix}_residuals.png"), bbox_inches="tight", pad_inches=0.01
        )
    plt.show()
    
    
def generate_log_residuals_figure(ys_real, preds_real, prefix, path, i):
    # Define colormap
    cm = plt.cm.get_cmap('viridis')
    my_cmap = cm(np.linspace(0,1,10))

    # Create new colormap
    my_cmap = ListedColormap(my_cmap)
    fig, axes = plt.subplots(2, 2, figsize=(10, 5))
    vmin = 0
    vmax=40
    axes[0,0].set_ylabel("Density", fontsize=13) 
    axes[1,0].set_ylabel("Residuals", fontsize=13)

    axes[0,0].grid()
    axes[1,0].grid()
    axes[0,1].grid()
    axes[1,1].grid()
    
    ys_real_ = np.log(ys_real)[np.log(preds_real)>-2]
    preds_real_ = np.log(preds_real)[np.log(preds_real)>-2]

    sns.kdeplot(data=ys_real_, ax=axes[0,0], label="True", color=my_cmap(1), fill=True)
    sns.kdeplot(data=preds_real_, ax=axes[0,0], label="Predicted", color='green', fill=True)
    axes[0,0].set_xlabel("Log of Chl-a concentration [log(mg/L)]", fontsize=13)
    axes[0,0].set_ylabel("Density", fontsize=13)
    axes[0,0].set_title("Log transformed", fontsize=15)
    axes[0,0].legend(fontsize=11)
    #axes[0,0].set_xlim(-0.5, 5.1)
    
    sns.kdeplot(data=ys_real, ax=axes[0,1], label="True", color=my_cmap(1), fill=True)
    sns.kdeplot(data=preds_real, ax=axes[0,1], label="Predicted", color='green', fill=True)
    
    axes[0,1].set_xlabel("Chl-a concentration \n [mg/m$^3$]", fontsize=13)
    axes[0,1].set_ylabel("Density", fontsize=13)
    axes[0,1].set_title("True scale", fontsize=15)
    axes[0,1].legend(fontsize=11)
    
    axes[1,0].scatter(ys_real_, np.subtract(preds_real_, ys_real_), alpha=0.2, 
                color=my_cmap(4))
    axes[1,0].set_xlabel("Observed", fontsize=13)
    
    axes[1,1].scatter(ys_real, np.subtract(preds_real, ys_real), alpha=0.2, 
                color=my_cmap(4))
    axes[1,1].set_xlabel("Observed", fontsize=13)
    if i == 0:
        axes[1,0].set_xlim(-5, 111)
        axes[1,1].set_xlim(-5, 111)
    if i != 'a':
        fig.suptitle(f"Day {i}", fontsize=15)
        axes[1,0].set_xlim(-5, 110)
        axes[1,1].set_xlim(-5, 110)
    # elif i == 'a':
    #     axes[0,1].set_xlim(-5, 160)
    #     axes[1,1].set_xlim(-5, 160)
        # axes[1,1].set_ylim(-120, 160)
    plt.tight_layout()
    
    
    plt.savefig(
            os.path.join(path, f"{prefix}_residuals.png"), bbox_inches="tight", pad_inches=0.01
        )
    plt.show()

#%%
    
ys_total = np.empty(())
preds_total = np.empty(())
for i in range(5):
    ys_real = np.empty(())
    preds_real = np.empty(())
    for p in range(len(all_pred_frames)):
        frames = all_pred_frames[p]
        original_frames = all_init_frames[p]
        frames = np.where(landmask == 0, np.nan, frames)
        original_frames = np.where(landmask == 0, np.nan, original_frames)

        frames = np.exp(frames)
        original_frames = np.exp(original_frames)
        
        ys_real = np.append(ys_real,original_frames[i+10,:,:].flatten())
        preds_real = np.append(preds_real, frames[i+10,:,:].flatten())
    #generate_pred_histogram(ys_real, preds_real, f"chl_{MODELNAME}_day_{i}", "./Figures", i+1) # 
    #generate_residuals_figure(ys_real, preds_real, f"chl_{MODELNAME}_day_{i}", "./Figures", i+1) 
    ys_total = np.append(ys_total, ys_real)
    preds_total = np.append(preds_total, preds_real)
    
#%%
generate_pred_histogram(np.log(ys_total), np.log(preds_total), f"chl_{MODELNAME}_day_total_log", "./Figures", 'a') #
generate_residuals_figure(ys_total, preds_total, f"chl_{MODELNAME}_day_total", "./Figures", 'a')

#%%
generate_log_residuals_figure(ys_total, preds_total, f"chl_{MODELNAME}_day_total_log", "./Figures", 'a')

#%%
# Feature importance analysis after https://www.kaggle.com/code/cdeotte/lstm-feature-importance
# To define whether a feature is important, the feature is shuffled randomly and the value predicted by the model is compared to the original value.
COLS = ['Chlorophyll-a','Cloudmask','Wind v10','Wind u10',
        'Air temperature', 'Precipitation', 'Radiation', 'Sea surface temperature', 
        'Wave VHM0', 'Wave VMDR', 'Topology', 'Sine of day of year', 'Cosine of day of year',
        'Longitude', 'Latitude']


results_1 = []
results_2 = []
results_3 = []
results_4 = []
results_5 = []
print(' Computing LSTM feature importance...')

# BASELINE (NO SHUFFLE)
baseline_rmses_1 =    np.mean(rmses[:,0])
baseline_rmses_2 =    np.mean(rmses[:,1])
baseline_rmses_3 =    np.mean(rmses[:,2])
baseline_rmses_4 =    np.mean(rmses[:,3])
baseline_rmses_5 =    np.mean(rmses[:,4])

results_1.append({'feature':'BASELINE','rmses':baseline_rmses_1})    
results_2.append({'feature':'BASELINE','rmses':baseline_rmses_2}) 
results_3.append({'feature':'BASELINE','rmses':baseline_rmses_3}) 
results_4.append({'feature':'BASELINE','rmses':baseline_rmses_4})    
results_5.append({'feature':'BASELINE','rmses':baseline_rmses_5})  

rmses_feat = np.empty((test_dataset[0].shape[-1], test_dataset[0].shape[0], target_size))   

# get the feature importance for each feature    
for k in tqdm.tqdm(range(test_dataset[0].shape[-1])):
    
    # SHUFFLE FEATURE K
    saved_dataset = test_dataset[0][:,:,:,k].copy()
    np.random.shuffle(test_dataset[0][:,:,:,k])
    
    for i in range(test_dataset[0].shape[0]):
        print(i)
        frames_input = test_dataset[0][i]
        frames_input = frames_input[:history_size, ...]
        
        example =test_dataset[1][i]
        
        frames = example[:history_size,:,:,0]
        original_frame = example[history_size:,:,:,0]
        
        print(original_frame.shape)
        
        print("Now on to the predictions")
        new_prediction = model.predict(np.expand_dims(frames_input, axis=0))
        
        # case single multi output
        predicted_frame = np.squeeze(new_prediction, axis=0)
        predicted_frame = np.expand_dims(predicted_frame[-1, ...], axis=0)
        predicted_frame = np.swapaxes(predicted_frame, 0, 3)
        print(predicted_frame.shape)
        
            
        # Extend the set of prediction frames.
        frames = np.concatenate((frames, predicted_frame[:,:,:,0]), axis=0)
        
        # transform to original scale
        predicted_frame = np.exp(predicted_frame)
        original_frame = np.exp(original_frame)
        
        for j in range(target_size):        
            # calculate the rmse and r2 score for the prediction per day
            rmses_feat[k,i,j] = np.sqrt(np.mean((original_frame[j,:,:] - predicted_frame[j,:,:,0])**2))

    # UNSHUFFLE FEATURE K
    test_dataset[0][:,:,:,k] = saved_dataset
    
    results_1.append({'feature':COLS[k],'rmses':np.mean(rmses_feat[k,:,0])})
    results_2.append({'feature':COLS[k],'rmses':np.mean(rmses_feat[k,:,1])})
    results_3.append({'feature':COLS[k],'rmses':np.mean(rmses_feat[k,:,2])})
    results_4.append({'feature':COLS[k],'rmses':np.mean(rmses_feat[k,:,3])})
    results_5.append({'feature':COLS[k],'rmses':np.mean(rmses_feat[k,:,4])})
    
#%%
# sort by rmses and plot the results
results_1 = pd.DataFrame(results_1).sort_values(by='rmses',ascending=True)
results_2 = pd.DataFrame(results_2).sort_values(by='rmses',ascending=True)
results_3 = pd.DataFrame(results_3).sort_values(by='rmses',ascending=True)
results_4 = pd.DataFrame(results_4).sort_values(by='rmses',ascending=True)
results_5 = pd.DataFrame(results_5).sort_values(by='rmses',ascending=True)

results = {
    0: results_1,
    1: results_2,
    2: results_3,
    3: results_4,
    4: results_5
}

baselines = {
    0: baseline_rmses_1,
    1: baseline_rmses_2,
    2: baseline_rmses_3,
    3: baseline_rmses_4,
    4: baseline_rmses_5
}

fig, axes = plt.subplots(1,5,figsize=(22,6))
for i in range(5):
    df = results[i]
    print(df)
    axes[i].barh(np.arange(len(COLS)+1),df.rmses, color="seagreen")
    axes[i].set_yticks(np.arange(len(COLS)+1),df.feature.values, size=12)
    axes[i].set_title(f'Day {i+1}',size=16)
    axes[i].set_ylim((-1,len(COLS)+1))
    axes[i].plot([baselines[i],baselines[i]],[-1,len(COLS)+1], '--', color='red',
                label=f'Baseline \nRMSE={baselines[i]:.3f}')
    axes[i].set_xlabel(f'RMSE',size=14)
    axes[i].legend(fontsize=11)
    axes[i].tick_params(axis='both', which='major', labelsize=12)
    axes[i].tick_params(axis='both', which='minor', labelsize=12)

axes[0].set_xlim((2.9,4.0))
axes[1].set_xlim((2.9,4.0))
axes[2].set_xlim((2.9,4.0))
axes[3].set_xlim((2.9,4.0))
axes[4].set_xlim((2.9,4.0))    
axes[0].set_ylabel('Permuted Feature',size=14)

plt.tight_layout()
plt.show()

#%%
# get the total feature importance for each feature
result = pd.concat(list(results.values()), axis=1)
result_ = result.drop(columns=['feature'])
result_ = result_.mean(axis=1)
result = result.loc[:, ~result.columns.duplicated()]
result['total_rmse'] = result_
result = result.sort_values(by='total_rmse',ascending=True)
print(result_)
print(result)

baseline = result[result.feature=='BASELINE']['total_rmse'].values[0]
print(baseline)
fix, axes = plt.subplots(1,1,figsize=(6,6))
axes.barh(np.arange(len(COLS)+1),result.total_rmse.values, color="seagreen")
axes.set_yticks(np.arange(len(COLS)+1),result.feature.values, size=12)
axes.set_title(f'Total Feature Importance',size=16)
axes.plot([baseline,baseline],
             [-1,len(COLS)+1], '--', color='red', label=f'Baseline \nRMSE={baseline:.3f}')
axes.set_ylim((-1,len(COLS)+1))
axes.set_xlabel(f'RMSE',size=14)
axes.tick_params(axis='both', which='major', labelsize=12)
axes.tick_params(axis='both', which='minor', labelsize=12)
axes.set_xlim((1,1.3))
axes.legend(fontsize=11)
plt.tight_layout()
plt.show()
    

#%%
plt.figure()
plt.imshow(frames[10,:,:]-frames[14,:,:])
plt.show()

#%%

# Select a few random examples from the dataset.
examples = test_dataset[np.random.choice(range(len(test_dataset)), size=5)]

# Iterate over the examples and predict the frames.
predicted_videos = []
for example in examples:
    # Pick the first/last ten frames from the example.
    frames = example[:10, ...]
    original_frames = example[10:, ...]
    new_predictions = np.zeros(shape=(10, *frames[0].shape))

    # Predict a new set of 10 frames.
    for i in range(10):
        # Extract the model's prediction and post-process it.
        frames = example[: 10 + i + 1, ...]
        new_prediction = model.predict(np.expand_dims(frames, axis=0))
        new_prediction = np.squeeze(new_prediction, axis=0)
        predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)

        # Extend the set of prediction frames.
        new_predictions[i] = predicted_frame

    # Create and save GIFs for each of the ground truth/prediction images.
    for frame_set in [original_frames, new_predictions]:
        # Construct a GIF from the selected video frames.
        print(frame_set.shape)
        frame_set = np.where(original_frames == 0, np.nan, frame_set )
        current_frames = np.squeeze(frame_set[:,:,:,0])
        current_frames = current_frames[..., np.newaxis] * np.ones(3)
        current_frames = (current_frames * np.nanmax(dataset)).astype(np.uint8)
        current_frames = list(current_frames)
        print(current_frames[0].shape)

        # Construct a GIF from the frames.
        with io.BytesIO() as gif:
            imageio.mimsave(gif, current_frames*255, "GIF", fps=8)
            predicted_videos.append(gif.getvalue())

# Display the videos.
print(" Truth\tPrediction")
for i in range(0, len(predicted_videos), 2):
    # Construct and display an `HBox` with the ground truth and prediction.
    box = HBox(
        [
            widgets.Image(value=predicted_videos[i]),
            widgets.Image(value=predicted_videos[i + 1]),
        ]
    )
    display(box)
# %%
