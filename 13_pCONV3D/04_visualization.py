#%%
import os
import sys
import numpy as np
import rasterio as rio
import time
import tensorflow as tf

from skimage import io as skio
from skimage import util as skutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import xarray as xr
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap


#%%

model_name_log = "model_203_RIGA_dataperc30-100_optimized"
DATA_PATH_LOG  = f"./training_results/prediction_{model_name_log}"

model_name_boxcox = "model_123_RIGA_dataperc30-100_optimized_2"
DATA_PATH_BOXCOX  = f"./training_results/prediction_{model_name_boxcox}"

model_name_minmax = "model_313_RIGA_dataperc30-100_optimized_2"
DATA_PATH_MINMAX  = f"./training_results/prediction_{model_name_minmax}"

file_number = "_139"
dataset = "multi"

if dataset == 'olci':
    splitting_array = np.arange(20, 2540, 20)
    time_array = pd.date_range(start='2016-05-01', end='2023-04-14', freq='D')
    time_split = np.split(time_array, splitting_array, axis=0)

    print(time_split[114])
else:
    splitting_array = np.arange(20, 7040, 20)
    time_array = pd.date_range(start='1997-09-04', end='2021-12-31', freq='D')
    time_split = np.split(time_array, splitting_array, axis=0)

    print(time_split[139])
    
#%%
file_name = [
            os.path.basename(os.path.join(DATA_PATH_LOG, fname))
            for fname in sorted(os.listdir(DATA_PATH_LOG))
            if fname.endswith(".tiff") and fname.startswith("PRED_")  #and 'olci' not in fname
            and file_number in fname
        ]

pred_files_log = [
        skio.imread(os.path.join(DATA_PATH_LOG, fname))
        for fname in sorted(os.listdir(DATA_PATH_LOG))
        if fname.endswith(".tiff") and fname.startswith("PRED_")  #and 'olci' not in fname
        and file_number in fname and dataset in fname
    ]
print("Min and max of Predictions log transformed")
print(np.nanmax(pred_files_log))
print(np.nanmin(pred_files_log))

y_files_log = [
        skio.imread(os.path.join(DATA_PATH_LOG, fname))
        for fname in sorted(os.listdir(DATA_PATH_LOG))
        if fname.endswith(".tiff") and fname.startswith("Y_")  #and 'olci' not in fname
        and file_number in fname
    ]
print("Min and max of Y files lof transformed")
print(np.nanmax(y_files_log))
print(np.nanmin(y_files_log))

pred_files_minmax = [
        skio.imread(os.path.join(DATA_PATH_MINMAX, fname))
        for fname in sorted(os.listdir(DATA_PATH_MINMAX))
        if fname.endswith(".tiff") and fname.startswith("PRED_")  #and 'olci' not in fname
        and file_number in fname and dataset in fname
    ]
print("Min and max of Predictions minmax transformed")
print(np.nanmax(pred_files_minmax))
print(np.nanmin(pred_files_minmax))

y_files_minmax = [
        skio.imread(os.path.join(DATA_PATH_MINMAX, fname))
        for fname in sorted(os.listdir(DATA_PATH_MINMAX))
        if fname.endswith(".tiff") and fname.startswith("Y_")  #and 'olci' not in fname
        and file_number in fname and dataset in fname
    ]
print("Min and max of Y files minmax transformed")
print(np.nanmax(y_files_minmax))
print(np.nanmin(y_files_minmax))

pred_files_bcx = [
        skio.imread(os.path.join(DATA_PATH_BOXCOX, fname))
        for fname in sorted(os.listdir(DATA_PATH_BOXCOX))
        if fname.endswith(".tiff") and fname.startswith("PRED_")  #and 'olci' not in fname
        and file_number in fname and dataset in fname
    ]
print("Min and max of Predictions bcx transformed")
print(np.nanmax(pred_files_bcx))
print(np.nanmin(pred_files_bcx))

y_files_bcx = [
        skio.imread(os.path.join(DATA_PATH_BOXCOX, fname))
        for fname in sorted(os.listdir(DATA_PATH_BOXCOX))
        if fname.endswith(".tiff") and fname.startswith("Y_")  #and 'olci' not in fname
        and file_number in fname and dataset in fname
    ]
print("Min and max of Y files bcx transformed")
print(np.nanmax(y_files_bcx))
print(np.nanmin(y_files_bcx))

valmask_file = [
            skio.imread(os.path.join(DATA_PATH_BOXCOX, fname))
            for fname in sorted(os.listdir(DATA_PATH_BOXCOX))
            if fname.endswith(".tiff") and fname.startswith("VALMASK_") #and 'olci' not in fname
            and file_number in fname and dataset in fname
        ][0]
mask_file = [
            skio.imread(os.path.join(DATA_PATH_BOXCOX, fname))
            for fname in sorted(os.listdir(DATA_PATH_BOXCOX))
            if fname.endswith(".tiff") and fname.startswith("MASK_") #and 'olci' not in fname
            and file_number in fname and dataset in fname
        ][0]
landmask_file = [
            skio.imread(os.path.join(DATA_PATH_BOXCOX, fname))
            for fname in sorted(os.listdir(DATA_PATH_BOXCOX))
            if fname.endswith(".tiff") and fname.startswith("LANDMASK_") #and 'olci' not in fname
            and file_number in fname and dataset in fname
        ][0]


if dataset == "multi":
    lambda_cb = -0.2816107171659248
    max_val = 194.84649658203125
    min_val = 0.12770438194274902
        
elif dataset == "olci":
    lambda_cb = -0.08346259836955258
    max_val = 171.9263916015625
    min_val = 1.000003457069397
    
preds_log_real = np.exp(pred_files_log[0])
ys_log_real = np.exp(y_files_log[0])
    
preds_minmax_real = pred_files_minmax[0] * (max_val - min_val) + min_val
ys_minmax_real = y_files_minmax[0] * (max_val - min_val) + min_val

preds_bcx_real = (pred_files_bcx[0]*lambda_cb + 1.0 )**(1/lambda_cb)
ys_bcx_real = (y_files_bcx[0]*lambda_cb + 1.0 )**(1/lambda_cb)

#%%# timestep
a = 15
vmax=40

y = np.where(mask_file == 1, ys_log_real, np.nan)
y = np.where(landmask_file == 1, y, np.nan)
y_ = np.where(valmask_file == 1, y, np.nan)
fig, axes = plt.subplots(1, 2, figsize=(5, 4))
axes[0].imshow(y[:,:,a], cmap='viridis', origin='lower', vmin=0, vmax=vmax)
axes[0].set_title('Real Data')
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[1].imshow(y_[:,:,a], cmap='viridis', origin='lower', vmin=0, vmax=vmax)
axes[1].set_title('Validation Pixels')
axes[1].set_xticks([])
axes[1].set_yticks([])
plt.show()
#%%
# Create a figure and a set of subplots

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
fig.suptitle('Comparison of different transformations')

pred_log = np.where(landmask_file==1,preds_log_real,np.nan)
pred_minmax = np.where(landmask_file==1,preds_minmax_real,np.nan)
pred_bcx = np.where(landmask_file==1,preds_bcx_real,np.nan)

axes[0].imshow(pred_log[:,:,a], cmap='viridis', origin='lower', vmin=0, vmax=40)
axes[0].set_title('Log transformed')
axes[0].set_ylabel('Predictions')
axes[0].set_yticks([])
axes[0].set_xticks([])
axes[1].imshow(pred_minmax[:,:,a], cmap='viridis', origin='lower', vmin=0, vmax=40)
axes[1].set_title('Minmax transformed')
axes[1].set_yticks([]) 
axes[1].set_xticks([])
axes[2].imshow(pred_bcx[:,:,a], cmap='viridis', origin='lower', vmin=0, vmax=40)
axes[2].set_title('Boxcox transformed')
axes[2].set_yticks([])
axes[2].set_xticks([])

plt.show()
# %%
def generate_pred_histogram(chl, preds, prefix, path, fig_height=4.5):
        sns.set_theme(style="whitegrid")
        print("Start generating histogram...")
        min_val = -0.01
        max_val = 0.4
        data = {"Original chl-a concentration [mg/L]": chl, "Predicted chl-a concentration [mg/L]": preds}

        g = sns.jointplot(
            x="Original chl-a concentration [mg/L]",
            y="Predicted chl-a concentration [mg/L]",
            kind="hist", # "kde"
            data=data,
            fill=True,
            color="g",
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
        plt.savefig(
            os.path.join(path, f"{prefix}_histogram_multi_139_bcx.png"), bbox_inches="tight", pad_inches=0.01
        )
        plt.show()

y_nr = np.where(mask_file == 1, y_files_minmax, np.nan)
y_nr = np.where(landmask_file == 1, y_nr, np.nan)
y_nr_ = np.where(valmask_file == 1, y_nr, np.nan)


pred_log_nr = np.where(landmask_file==1,pred_files_log,np.nan)
pred_minmax_nr = np.where(landmask_file==1,pred_files_minmax,np.nan)
pred_bcx_nr = np.where(landmask_file==1,pred_files_bcx,np.nan)

pred_log_nr = np.where(mask_file==1,pred_log_nr,np.nan)
pred_minmax_nr = np.where(mask_file==1,pred_minmax_nr,np.nan)
pred_bcx_nr = np.where(mask_file==1,pred_bcx_nr,np.nan)

pred_log = np.where(mask_file==1,pred_log,np.nan)
pred_minmax = np.where(mask_file==1,pred_minmax,np.nan)
pred_bcx = np.where(mask_file==1,pred_bcx,np.nan)



pred_log_val = np.where(valmask_file == 1, pred_log, np.nan)
pred_minmax_val = np.where(valmask_file == 1, pred_minmax, np.nan)
pred_bcx_val = np.where(valmask_file == 1, pred_bcx, np.nan) 

pred_log_val_nr = np.where(valmask_file == 1, pred_log_nr, np.nan)
pred_minmax_val_nr = np.where(valmask_file == 1, pred_minmax_nr, np.nan)
pred_bcx_val_nr = np.where(valmask_file == 1, pred_bcx_nr, np.nan) 

#%%
   
generate_pred_histogram(y_nr_.flatten(), pred_bcx_val_nr.flatten(), 'trans', path="../")

# %%

print("Plotting the Residuals")
fig, ax = plt.subplots(1,4, figsize=(15,4))

# Define colormap
cm = plt.cm.get_cmap('viridis')
my_cmap = cm(np.linspace(0,1,10))

# Create new colormap
my_cmap = ListedColormap(my_cmap)

y_nr = np.where(mask_file == 1, y_files_bcx, np.nan)
y_nr = np.where(landmask_file == 1, y_nr, np.nan)
y_nr_ = np.where(valmask_file == 1, y_nr, np.nan)
ax[0].scatter(y_nr_, np.subtract(pred_bcx_val_nr,y_nr_), alpha=0.8, 
              label="Boxcox transformed", marker='x', color=my_cmap(0))

y_nr = np.where(mask_file == 1, y_files_minmax, np.nan)
y_nr = np.where(landmask_file == 1, y_nr, np.nan)
y_nr_ = np.where(valmask_file == 1, y_nr, np.nan)
ax[1].scatter(y_nr_, np.subtract(pred_minmax_val_nr,y_nr_), alpha=0.8, 
              label="Minmax transformed", marker="D", color='none', edgecolor=my_cmap(3))

y_nr = np.where(mask_file == 1, y_files_log, np.nan)
y_nr = np.where(landmask_file == 1, y_nr, np.nan)
y_nr_ = np.where(valmask_file == 1, y_nr, np.nan)
ax[2].scatter(y_nr_, np.subtract(pred_log_val_nr,y_nr_), alpha=0.8, 
              label="Log transformed", marker="o", color='none', edgecolor=my_cmap(8))
ax[0].set_ylabel("Residuals [transformed mg/m$^{3}$]", fontsize=12)
ax[0].set_xlabel("True values [transformed mg/m$^{3}$]", fontsize=12)
ax[0].set_title("Boxcox Transformed", fontsize=14)

ax[1].set_ylabel("Residuals [transformed mg/m$^{3}$]", fontsize=12)
ax[1].set_xlabel("True values [transformed mg/m$^{3}$]", fontsize=12)
ax[1].set_title("Minmax Transformed", fontsize=14)

ax[2].set_ylabel("Residuals [transformed mg/m$^{3}$]", fontsize=12)
ax[2].set_xlabel("True values [transformed mg/m$^{3}$]", fontsize=12)
ax[2].set_title("Log Transformed",  fontsize=14)

ax[3].scatter(y_, np.subtract(pred_bcx_val, y_), alpha=0.8, 
              label="Boxcox", marker='x', color=my_cmap(0))
ax[3].scatter(y_, np.subtract(pred_minmax_val, y_), alpha=0.8, 
              label="Minmax", marker="D", color='none', edgecolor=my_cmap(3))
ax[3].scatter(y_, np.subtract(pred_log_val, y_), alpha=0.8, 
              label="Log", marker="o", color='none', edgecolor=my_cmap(8))
ax[3].set_ylabel("Residuals [mg/m$^{3}$]", fontsize=12)
ax[3].set_xlabel("True values [mg/m$^{3}$]", fontsize=12)
ax[3].set_title("Back transformed to true scale",   fontsize=14)

#fig.suptitle("Residuals of the validation set", fontsize=16)    
plt.legend()
plt.tight_layout()
plt.show()
# %%

# Plots that include all three models
# DINCAE
model_name_DINCAE = "DINCAE_1_1_0_0_0_0_1_0_RIGA_dataperc20-100"
PATH_DINCAE = "../DINCAE_python/data/test_data"
DATA_PATH_PREDICTION_DINCAE = f"../DINCAE_python/data/prediction_{model_name_DINCAE}_GapSizeVar"

file_name = "139"

DINCAE_preds = [
        np.exp(xr.open_dataset(os.path.join(DATA_PATH_PREDICTION_DINCAE, fname))['mean_rec'].values/10)
        for fname in sorted(os.listdir(DATA_PATH_PREDICTION_DINCAE))
        if fname.endswith(".nc") and fname.startswith("PRED_") 
        and file_name in fname
    ]

ys = [
        np.exp(xr.open_dataset(os.path.join(DATA_PATH_PREDICTION_DINCAE, fname))['mean_rec'].values/10)
        for fname in sorted(os.listdir(DATA_PATH_PREDICTION_DINCAE))
        if fname.endswith(".nc") and fname.startswith("Y_")  
        and file_name in fname
    ]

xs = [
        xr.open_dataset(os.path.join(DATA_PATH_PREDICTION_DINCAE, fname))['mean_rec'].values
        for fname in sorted(os.listdir(DATA_PATH_PREDICTION_DINCAE))
        if fname.endswith(".nc") and fname.startswith("X_")  
        and file_name in fname
    ]

# DINpCAE
model_name_DINpCAE = "pDINCAE_1_1_1_0_1_0_2_A_T_S_S_dataperc30-100"
PATH_DINpCAE = "../DINCAE_python/data/test_data"
DATA_PATH_PREDICTION_DINpCAE = f"../DINpCAE_python/data/prediction_{model_name_DINpCAE}_GapSizeVar"

DINpCAE_preds = [
        np.exp(xr.open_dataset(os.path.join(DATA_PATH_PREDICTION_DINpCAE, fname))['mean_rec'].values/10)
        for fname in sorted(os.listdir(DATA_PATH_PREDICTION_DINpCAE))
        if fname.endswith(".nc") and fname.startswith("PRED_") 
        and file_name in fname
    ]

# pCONV3D
model_name_pCONV3D = "model_203_RIGA_dataperc30-100_optimized"
DATA_PATH_PREDICTION_pCONV3D  = f"./training_results/prediction_{model_name_pCONV3D}_GapSizeVar"

pCONV3D_preds = [
        np.exp(skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D, fname)))
        for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D))
        if fname.endswith(".tiff") and fname.startswith("PRED_")  #and 'olci' not in fname
        and file_name in fname 
    ]

valmask_file = [
            skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D, fname))
            for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D))
            if fname.endswith(".tiff") and fname.startswith("VALMASK_") #and 'olci' not in fname
            and file_name in fname 
        ][0]
mask_file = [
            skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D, fname))
            for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D))
            if fname.endswith(".tiff") and fname.startswith("MASK_") #and 'olci' not in fname
            and file_name in fname 
        ][0]
landmask_file = [
            skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D, fname))
            for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D))
            if fname.endswith(".tiff") and fname.startswith("LANDMASK_") #and 'olci' not in fname
            and file_name in fname 
        ][0]

for i in range(len(pCONV3D_preds)):
    pCONV3D_preds[i][landmask_file==0] = np.nan

#%% 
fig, axes = plt.subplots(4, 6, figsize=(12, 6))
d = 3
vmin = 0
vmax=40
axes[0,0].set_ylabel("Input data")
axes[1,0].set_ylabel("DINpCAE")
axes[2,0].set_ylabel("DINCAE") 
axes[3,0].set_ylabel("pCONV3D") 

axes[0,0].set_title("45% missing")
axes[0,1].set_title("55% missing")
axes[0,2].set_title("65% missing")
axes[0,3].set_title("75% missing")
axes[0,4].set_title("85% missing")
axes[0,5].set_title("95% missing")
for a in range(6):
    im = axes[0,a].imshow(np.where(np.isnan(xs[a][d,:,:]), np.nan, ys[a][d,:,:]), 
                          vmin=vmin, vmax=vmax, origin='lower')
    axes[0,a].set_xticks([])
    axes[0,a].set_yticks([])
    axes[1,a].imshow(DINpCAE_preds[a][d,:,:], vmin=vmin, vmax=vmax, origin='lower')
    axes[1,a].set_xticks([])
    axes[1,a].set_yticks([])
    axes[2,a].imshow(DINCAE_preds[a][d,:,:], vmin=vmin, vmax=vmax, origin='lower')
    axes[2,a].set_xticks([])
    axes[2,a].set_yticks([])
    axes[3,a].imshow(pCONV3D_preds[a][:,:,d], vmin=vmin, vmax=vmax, origin='lower')
    axes[3,a].set_xticks([])
    axes[3,a].set_yticks([])
plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="Chl-a concentration [mg/m$^3$]", pad=0.01)
#plt.tight_layout()
plt.show()


# %%

# Visualization of gap filled data for all tree areas 
# Distribution, input data, real data, error, gap filled data

# multi 
file_name = "92" #GTL: 340, KTT: 378, RIGA: 139

# olci
#file_name = "92" # RIGA: 114, KTT: 60, GTL: 92
# Load data
model_name_pCONV3D = "model_213_GTL_dataperc30-100_3"
DATA_PATH_PREDICTION_pCONV3D  = f"./training_results/prediction_{model_name_pCONV3D}"

pCONV3D_preds = [
        np.exp(skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D, fname)))
        for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D))
        if fname.endswith(".tiff") and fname.startswith("PRED_")  #and 'olci' not in fname
        and file_name in fname 
    ][0]
x_files = [
        np.exp(skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D, fname)))
        for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D))
        if fname.endswith(".tiff") and fname.startswith("X_") 
        and file_name  in fname
    ][0]

y_files = [
        np.exp(skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D, fname)))
        for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D))
        if fname.endswith(".tiff") and fname.startswith("Y_")  #and 'olci' not in fname
        and file_name  in fname
    ][0]

valmask_file = [
            skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D, fname))
            for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D))
            if fname.endswith(".tiff") and fname.startswith("VALMASK_") #and 'olci' not in fname
            and file_name in fname 
        ][0]
mask_file = [
            skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D, fname))
            for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D))
            if fname.endswith(".tiff") and fname.startswith("MASK_") #and 'olci' not in fname
            and file_name in fname 
        ][0]
landmask_file = [
            skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D, fname))
            for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D))
            if fname.endswith(".tiff") and fname.startswith("LANDMASK_") #and 'olci' not in fname
            and file_name in fname 
        ][0]

for i in range(len(pCONV3D_preds)):
    pCONV3D_preds[landmask_file==0] = np.nan
    
#%%
# Define colormap
cm = plt.cm.get_cmap('viridis')
my_cmap = cm(np.linspace(0,1,10))

# Create new colormap
my_cmap = ListedColormap(my_cmap)
#%%
print(valmask_file.shape)
fig, axes = plt.subplots(4, 3, figsize=(5.4, 5.8))
d = 3
vmin = 0
vmax=5 # 5
axes[0,0].set_ylabel("True data", fontsize=13)
axes[1,0].set_ylabel("Input data", fontsize=13)
axes[2,0].set_ylabel("Prediction", fontsize=13)
axes[3,0].set_ylabel("Difference", fontsize=13)

axes[0,0].set_title("(D)", fontsize=13)
axes[0,1].set_title("(E)", fontsize=13)
axes[0,2].set_title("(F)",  fontsize=13)
# axes[0,3].set_title("(p)")
# axes[0,4].set_title("(q)")
# axes[0,5].set_title("(r)")
for a in range(3):
    print(a)
    d = a + 4  # 4 GTL, 11 KTT, 10 RIGA
    #axes[0,a].imshow(np.where(landmask_file[:,:,d]==0,1,np.nan), vmin=0, vmax=1, cmap='Greys',origin='lower')
    axes[0,a].set_xticks([])
    im = axes[0,a].imshow(y_files[:,:,d], vmin=vmin, vmax=vmax, origin='lower')
    axes[0,a].set_yticks([])
    #axes[1,a].imshow(np.where(landmask_file[:,:,d]==0,1,np.nan), vmin=0, vmax=1, cmap='Greys',origin='lower')
    axes[1,a].imshow(np.where(np.isnan(x_files[:,:,d]), np.nan, y_files[:,:,d]), 
                          vmin=vmin, vmax=vmax, origin='lower')
    axes[1,a].set_xticks([])
    axes[1,a].set_yticks([])
    #axes[2,a].imshow(np.where(landmask_file[:,:,d]==0,1,np.nan), vmin=0, vmax=1, cmap='Greys',origin='lower')
    axes[2,a].imshow(pCONV3D_preds[:,:,d], vmin=vmin, vmax=vmax, origin='lower')
    axes[2,a].set_xticks([])
    axes[2,a].set_yticks([])
    #axes[3,a].imshow(np.where(landmask_file[:,:,d]==0,1,np.nan), vmin=0, vmax=1, cmap='Greys',origin='lower')
    im2 = axes[3,a].imshow(pCONV3D_preds[:,:,d]-y_files[:,:,d], vmin=-5, vmax=5, origin='lower', cmap='RdYlGn')
    axes[3,a].set_xticks([])
    axes[3,a].set_yticks([])
    
plt.colorbar(im, ax=axes[:3,:].ravel().tolist(), shrink=0.8, label="Chl-a concentration [mg/m$^3$]", pad=0.02, aspect=25)
plt.colorbar(im2, ax=axes[3:,:].ravel().tolist(), shrink=0.6, label="Error [mg/m$^3$]", pad=0.02, aspect=7)
#plt.tight_layout()
plt.show()
# %%
import seaborn as sns
fig, axes = plt.subplots(2, 3, figsize=(7, 5), sharey = 'row')
d = 3
vmin = 0
vmax=40
axes[0,0].set_ylabel("Density", fontsize=13) 
axes[1,0].set_ylabel("Residuals", fontsize=13)

axes[0,0].set_title("(g)",  fontsize=13)
axes[0,1].set_title("(h)", fontsize=13)
axes[0,2].set_title("(i)", fontsize=13)

for a in range(3):
    axes[0, a].grid()
    axes[1, a].grid()
    d = a + 4  # 11 KTT, 10 RIGA
    val_preds = np.where(valmask_file[:, :, d] == 0, np.nan, pCONV3D_preds[:, :, d])
    val_ys = np.where(valmask_file[:, :, d] == 0, np.nan, y_files[:, :, d])

    pred_graph_ = val_preds[~np.isnan(val_preds)]
    ys_graph_ = val_ys[~np.isnan(val_preds)]
    pred_graph = pred_graph_[~np.isnan(ys_graph_)]
    ys_graph = ys_graph_[~np.isnan(ys_graph_)]

    # Plot kernel density functions
    sns.kdeplot(data=ys_graph, ax=axes[0, a], label="True", color=my_cmap(1), fill=True)
    sns.kdeplot(data=pred_graph, ax=axes[0, a], label="Predicted", color='green', fill=True)
    
    axes[0, a].set_xlabel("Chl-a concentration \n [mg/m$^3$]", fontsize=13)
    axes[0, a].legend(fontsize=11)

    axes[1, a].scatter(val_ys, np.subtract(val_preds, val_ys), alpha=0.2, color=my_cmap(4))
    axes[1, a].set_xlabel("Observed", fontsize=13)

plt.tight_layout()
plt.show()
    

plt.tight_layout()
# %%
# plot a time series of gap-filled images for all three areas
# multi 
file_name_RIGA = "114" 
file_name_KTT = "60" 
file_name_GTL = "92" 

# olci
#file_name = "92" # RIGA: 114, KTT: 60, GTL: 92
# Load data
model_name_pCONV3D_RIGA = "model_213_RIGA_dataperc30-100"
DATA_PATH_PREDICTION_pCONV3D_RIGA  = f"./training_results/prediction_{model_name_pCONV3D_RIGA}"

pCONV3D_preds_RIGA = [
        np.exp(skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D_RIGA, fname)))
        for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D_RIGA))
        if fname.endswith(".tiff") and fname.startswith("PRED_")  #and 'olci' not in fname
        and file_name_RIGA in fname 
    ][0]
x_files_RIGA = [
        np.exp(skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D_RIGA, fname)))
        for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D_RIGA))
        if fname.endswith(".tiff") and fname.startswith("X_") 
        and file_name_RIGA  in fname
    ][0]

y_files_RIGA = [
        np.exp(skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D_RIGA, fname)))
        for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D_RIGA))
        if fname.endswith(".tiff") and fname.startswith("Y_")  #and 'olci' not in fname
        and file_name_RIGA  in fname
    ][0]

valmask_file_RIGA = [
            skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D_RIGA, fname))
            for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D_RIGA))
            if fname.endswith(".tiff") and fname.startswith("VALMASK_") #and 'olci' not in fname
            and file_name_RIGA in fname 
        ][0]
mask_file_RIGA = [
            skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D_RIGA, fname))
            for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D_RIGA))
            if fname.endswith(".tiff") and fname.startswith("MASK_") #and 'olci' not in fname
            and file_name_RIGA in fname 
        ][0]
landmask_file_RIGA = [
            skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D_RIGA, fname))
            for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D_RIGA))
            if fname.endswith(".tiff") and fname.startswith("LANDMASK_") #and 'olci' not in fname
            and file_name_RIGA in fname 
        ][0]

model_name_pCONV3D_KTT = "model_213_KTT_dataperc30-100_3"
DATA_PATH_PREDICTION_pCONV3D_KTT  = f"./training_results/prediction_{model_name_pCONV3D_KTT}"

pCONV3D_preds_KTT = [
        np.exp(skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D_KTT, fname)))
        for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D_KTT))
        if fname.endswith(".tiff") and fname.startswith("PRED_")  #and 'olci' not in fname
        and file_name_KTT in fname 
    ][0]
x_files_KTT = [
        np.exp(skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D_KTT, fname)))
        for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D_KTT))
        if fname.endswith(".tiff") and fname.startswith("X_") 
        and file_name_KTT  in fname
    ][0]

y_files_KTT = [
        np.exp(skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D_KTT, fname)))
        for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D_KTT))
        if fname.endswith(".tiff") and fname.startswith("Y_")  #and 'olci' not in fname
        and file_name_KTT  in fname
    ][0]

valmask_file_KTT = [
            skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D_KTT, fname))
            for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D_KTT))
            if fname.endswith(".tiff") and fname.startswith("VALMASK_") #and 'olci' not in fname
            and file_name_KTT in fname 
        ][0]
mask_file_KTT = [
            skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D_KTT, fname))
            for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D_KTT))
            if fname.endswith(".tiff") and fname.startswith("MASK_") #and 'olci' not in fname
            and file_name_KTT in fname 
        ][0]
landmask_file_KTT = [
            skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D_KTT, fname))
            for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D_KTT))
            if fname.endswith(".tiff") and fname.startswith("LANDMASK_") #and 'olci' not in fname
            and file_name_KTT in fname 
        ][0]

for i in range(len(pCONV3D_preds_KTT)):
    pCONV3D_preds_KTT[landmask_file_KTT==0] = np.nan
    
model_name_pCONV3D_GTL = "model_213_GTL_dataperc30-100_4"
DATA_PATH_PREDICTION_pCONV3D_GTL  = f"./training_results/prediction_{model_name_pCONV3D_GTL}"

pCONV3D_preds_GTL = [
        np.exp(skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D_GTL, fname)))
        for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D_GTL))
        if fname.endswith(".tiff") and fname.startswith("PRED_")  #and 'olci' not in fname
        and file_name_GTL in fname 
    ][0]
x_files_GTL = [
        np.exp(skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D_GTL, fname)))
        for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D_GTL))
        if fname.endswith(".tiff") and fname.startswith("X_") 
        and file_name_GTL  in fname
    ][0]

y_files_GTL = [
        np.exp(skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D_GTL, fname)))
        for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D_GTL))
        if fname.endswith(".tiff") and fname.startswith("Y_")  #and 'olci' not in fname
        and file_name_GTL  in fname
    ][0]

valmask_file_GTL = [
            skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D_GTL, fname))
            for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D_GTL))
            if fname.endswith(".tiff") and fname.startswith("VALMASK_") #and 'olci' not in fname
            and file_name_GTL in fname 
        ][0]
mask_file_GTL = [
            skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D_GTL, fname))
            for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D_GTL))
            if fname.endswith(".tiff") and fname.startswith("MASK_") #and 'olci' not in fname
            and file_name_GTL in fname 
        ][0]
landmask_file_GTL = [
            skio.imread(os.path.join(DATA_PATH_PREDICTION_pCONV3D_GTL, fname))
            for fname in sorted(os.listdir(DATA_PATH_PREDICTION_pCONV3D_GTL))
            if fname.endswith(".tiff") and fname.startswith("LANDMASK_") #and 'olci' not in fname
            and file_name_GTL in fname 
        ][0]

for i in range(len(pCONV3D_preds_GTL)):
    pCONV3D_preds_GTL[landmask_file_GTL==0] = np.nan

# %%
# show timeseries of 20 days with gaps filled for all three areas
fig, ax = plt.subplots(5,4, figsize=(10,13))

for i in range(4):
    ax[0,i].imshow(y_files_RIGA[:,:,i],vmin=0, vmax=40, origin='lower')
    ax[1,i].imshow(y_files_RIGA[:,:,i+4], vmin=0, vmax=40, origin='lower')
    ax[2,i].imshow(y_files_RIGA[:,:,i+8], vmin=0, vmax=40, origin='lower')
    ax[3,i].imshow(y_files_RIGA[:,:,i+12], vmin=0, vmax=40, origin='lower')
    ax[4,i].imshow(y_files_RIGA[:,:,i+16], vmin=0, vmax=40, origin='lower')

    ax[0,i].set_xticks([])
    ax[0,i].set_yticks([])
    ax[1,i].set_xticks([])
    ax[1,i].set_yticks([])
    ax[2,i].set_xticks([])
    ax[2,i].set_yticks([])
    ax[3,i].set_xticks([])
    ax[3,i].set_yticks([])
    ax[4,i].set_xticks([])
    ax[4,i].set_yticks([])
fig.subtitle('')

#%%
import matplotlib.gridspec as gridspec


def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        #ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False, length=0)


# gridspec inside gridspec
fig = plt.figure(figsize=(15, 5.5))

gs0 = gridspec.GridSpec(1, 2, figure=fig)

gs00 = gridspec.GridSpecFromSubplotSpec(4, 5, subplot_spec=gs0[0], wspace=0.05, 
                                        hspace=0.05)

for i in range(5):
    ax1 = fig.add_subplot(gs00[0, i])
    ax2 = fig.add_subplot(gs00[1, i])
    ax3 = fig.add_subplot(gs00[2, i])
    ax4 = fig.add_subplot(gs00[3, i])
    ax1.imshow(y_files_RIGA[:,:,i],vmin=0, vmax=40, origin='lower')
    ax2.imshow(y_files_RIGA[:,:,i+5], vmin=0, vmax=40, origin='lower')
    ax3.imshow(y_files_RIGA[:,:,i+10], vmin=0, vmax=40, origin='lower')
    ax4.imshow(y_files_RIGA[:,:,i+15], vmin=0, vmax=40, origin='lower')


# the following syntax does the same as the GridSpecFromSubplotSpec call above:
gs01 = gs0[1].subgridspec(4, 5, wspace=0.05, hspace=0.05)

for i in range(5):
    
    ax6 = fig.add_subplot(gs01[0, i])
    ax7 = fig.add_subplot(gs01[1, i])
    ax8 = fig.add_subplot(gs01[2, i])
    ax9 = fig.add_subplot(gs01[3, i])
    ax6.imshow(np.where(np.isnan(y_files_RIGA[:,:,i]),pCONV3D_preds_RIGA[:,:,i],y_files_RIGA[:,:,i]), 
                vmin=0, vmax=40, origin='lower')
    ax7.imshow(np.where(np.isnan(y_files_RIGA[:,:,i+5]),pCONV3D_preds_RIGA[:,:,i+5],y_files_RIGA[:,:,i+5]), 
                   vmin=0, vmax=40, origin='lower')
    ax8.imshow(np.where(np.isnan(y_files_RIGA[:,:,i+10]),pCONV3D_preds_RIGA[:,:,i+10],y_files_RIGA[:,:,i+10]), 
                   vmin=0, vmax=40, origin='lower')
    ax9.imshow(np.where(np.isnan(y_files_RIGA[:,:,i+15]),pCONV3D_preds_RIGA[:,:,i+15],y_files_RIGA[:,:,i+15]), 
                   vmin=0, vmax=40, origin='lower')
    
ax_a = fig.add_subplot(gs0[0])
ax_a.axis('off')
ax_a.set_title('Original Data', fontsize=16)

ax_b = fig.add_subplot(gs0[1])
ax_b.axis('off')
ax_b.set_title('Gap Filled Data', fontsize=16)

plt.suptitle("Gulf of Riga", fontsize=20)
format_axes(fig)
plt.show()
#%%
    

def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        #ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False, length=0)


# gridspec inside gridspec
fig = plt.figure(figsize=(15, 5.5))

gs0 = gridspec.GridSpec(1, 2, figure=fig)

gs00 = gridspec.GridSpecFromSubplotSpec(4, 5, subplot_spec=gs0[0], wspace=0.05, 
                                        hspace=0.05)

for i in range(5):
    ax1 = fig.add_subplot(gs00[0, i])
    ax2 = fig.add_subplot(gs00[1, i])
    ax3 = fig.add_subplot(gs00[2, i])
    ax4 = fig.add_subplot(gs00[3, i])
    ax1.imshow(y_files_KTT[:,:,i],vmin=0, vmax=5, origin='lower')
    ax2.imshow(y_files_KTT[:,:,i+5], vmin=0, vmax=5, origin='lower')
    ax3.imshow(y_files_KTT[:,:,i+10], vmin=0, vmax=5, origin='lower')
    ax4.imshow(y_files_KTT[:,:,i+15], vmin=0, vmax=5, origin='lower')


# the following syntax does the same as the GridSpecFromSubplotSpec call above:
gs01 = gs0[1].subgridspec(4, 5, wspace=0.05, hspace=0.05)

for i in range(5):
    
    ax6 = fig.add_subplot(gs01[0, i])
    ax7 = fig.add_subplot(gs01[1, i])
    ax8 = fig.add_subplot(gs01[2, i])
    ax9 = fig.add_subplot(gs01[3, i])
    ax6.imshow(np.where(np.isnan(y_files_KTT[:,:,i]),pCONV3D_preds_KTT[:,:,i],y_files_KTT[:,:,i]), 
                vmin=0, vmax=5, origin='lower')
    ax7.imshow(np.where(np.isnan(y_files_KTT[:,:,i+5]),pCONV3D_preds_KTT[:,:,i+5],y_files_KTT[:,:,i+5]), 
                   vmin=0, vmax=5, origin='lower')
    ax8.imshow(np.where(np.isnan(y_files_KTT[:,:,i+10]),pCONV3D_preds_KTT[:,:,i+10],y_files_KTT[:,:,i+10]), 
                   vmin=0, vmax=5, origin='lower')
    ax9.imshow(np.where(np.isnan(y_files_KTT[:,:,i+15]),pCONV3D_preds_KTT[:,:,i+15],y_files_KTT[:,:,i+15]), 
                   vmin=0, vmax=5, origin='lower')
    
ax_a = fig.add_subplot(gs0[0])
ax_a.axis('off')
ax_a.set_title('Original Data', fontsize=16)

ax_b = fig.add_subplot(gs0[1])
ax_b.axis('off')
ax_b.set_title('Gap Filled Data', fontsize=16)

plt.suptitle("Southern Kattegat", fontsize=20)
format_axes(fig)
plt.show()

# %%
def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        #ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False, length=0)


# gridspec inside gridspec
fig = plt.figure(figsize=(15, 5.5))

gs0 = gridspec.GridSpec(1, 2, figure=fig)

gs00 = gridspec.GridSpecFromSubplotSpec(4, 5, subplot_spec=gs0[0], wspace=0.05, 
                                        hspace=0.05)

for i in range(5):
    ax1 = fig.add_subplot(gs00[0, i])
    ax2 = fig.add_subplot(gs00[1, i])
    ax3 = fig.add_subplot(gs00[2, i])
    ax4 = fig.add_subplot(gs00[3, i])
    ax1.imshow(y_files_GTL[:,:,i],vmin=0, vmax=8, origin='lower')
    ax2.imshow(y_files_GTL[:,:,i+5], vmin=0, vmax=8, origin='lower')
    ax3.imshow(y_files_GTL[:,:,i+10], vmin=0, vmax=8, origin='lower')
    ax4.imshow(y_files_GTL[:,:,i+15], vmin=0, vmax=8, origin='lower')


# the following syntax does the same as the GridSpecFromSubplotSpec call above:
gs01 = gs0[1].subgridspec(4, 5, wspace=0.05, hspace=0.05)

for i in range(5):
    
    ax6 = fig.add_subplot(gs01[0, i])
    ax7 = fig.add_subplot(gs01[1, i])
    ax8 = fig.add_subplot(gs01[2, i])
    ax9 = fig.add_subplot(gs01[3, i])
    ax6.imshow(np.where(np.isnan(y_files_GTL[:,:,i]),pCONV3D_preds_GTL[:,:,i],y_files_GTL[:,:,i]), 
                vmin=0, vmax=8, origin='lower')
    ax7.imshow(np.where(np.isnan(y_files_GTL[:,:,i+5]),pCONV3D_preds_GTL[:,:,i+5],y_files_GTL[:,:,i+5]), 
                   vmin=0, vmax=8, origin='lower')
    ax8.imshow(np.where(np.isnan(y_files_GTL[:,:,i+10]),pCONV3D_preds_GTL[:,:,i+10],y_files_GTL[:,:,i+10]), 
                   vmin=0, vmax=8, origin='lower')
    ax9.imshow(np.where(np.isnan(y_files_GTL[:,:,i+15]),pCONV3D_preds_GTL[:,:,i+15],y_files_GTL[:,:,i+15]), 
                   vmin=0, vmax=8, origin='lower')
    
ax_a = fig.add_subplot(gs0[0])
ax_a.axis('off')
ax_a.set_title('Original Data', fontsize=16)

ax_b = fig.add_subplot(gs0[1])
ax_b.axis('off')
ax_b.set_title('Gap Filled Data', fontsize=16)

plt.suptitle("Gotland Basin", fontsize=20)
format_axes(fig)
plt.show()
# %%
