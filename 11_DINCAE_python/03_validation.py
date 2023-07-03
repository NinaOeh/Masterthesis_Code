'''
Script for validation of the DINCAE model.
'''

#%%
%matplotlib ipympl
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
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp


model_name = "DINCAE_1_1_0_0_0_0_1_0_RIGA_dataperc20-100"
PATH = "../data/test_data"
DATA_PATH_PREDICTION = f"../data/prediction_{model_name}_GapSizeVar"


#%%
def mean_squared_error(y_true, y_pred):    
    
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    
    if y_pred.shape[0] != 0 and y_true.shape[0] != 0:
            print("Max values of y_pred and y_true:")
            print(np.nanmax(y_pred))
            print(np.nanmax(y_true))
    else:
            print("No data in file")
    
    y_true_new_ = y_true[~np.isnan(y_pred)]
    y_true_new = y_true_new_[~np.isnan(y_true_new_)]
    y_pred_new_ = y_pred[~np.isnan(y_true)]
    y_pred_new = y_pred_new_[~np.isnan(y_pred_new_)]
    
    # subtract y_true from y_pred ignoring nan values
    subtracted = np.subtract(y_true_new, y_pred_new)
    mse = np.mean((subtracted)**2)
    return mse

def spatial_correlation_coefficient(y_true, y_pred):
    sccs = []
    for i in range(y_true.shape[0]):
        scc_value = scc(y_pred[i,:,:], y_true[i,:,:])
        sccs.append(scc_value)
    return np.mean(sccs)

def mse_gaps():
    """
    Computes MSE for gap values only
    """
    def loss(y_true, y_pred, y_train):      
        
        y_pred_val = np.where(~np.isnan(y_true), y_pred, np.nan)
        y_pred_val = np.where(np.isnan(y_train), y_pred_val, np.nan)
    
        
        y_true_val = np.where(np.isnan(y_train), y_true, np.nan)
        y_true_val = np.where(~np.isnan(y_pred_val), y_true_val, np.nan)
        
        
        print(f"Real pred max: {np.nanmax(y_pred)}")
        print(f"Real max: {np.nanmax(y_true)}")
        
        print(f"Real pred max, validation: {np.nanmax(y_pred_val)}")
        print(f"Real max, validation: {np.nanmax(y_true_val)}")
        
        mse = mean_squared_error(y_true_val, y_pred_val)
        scc_value = spatial_correlation_coefficient(y_true, y_pred)
        return mse, scc_value
    return loss

missing_perc = "_95_"

file_names = [
            os.path.basename(os.path.join(DATA_PATH_PREDICTION, fname))
            for fname in sorted(os.listdir(DATA_PATH_PREDICTION))
            if fname.endswith(".nc") and fname.startswith("PRED_")  
            and missing_perc in fname
        ]
print(f"Pred File names: {file_names}")

file_numbers = [file_name.split(".")[0] for file_name in file_names]


pred_files_trans = [
        xr.open_dataset(os.path.join(DATA_PATH_PREDICTION, fname))['mean_rec'].values
        for fname in sorted(os.listdir(DATA_PATH_PREDICTION))
        if fname.endswith(".nc") and fname.startswith("PRED_") 
        and missing_perc in fname
    ]

std_files_trans = [
        xr.open_dataset(os.path.join(DATA_PATH_PREDICTION, fname))['sigma_rec'].values
        for fname in sorted(os.listdir(DATA_PATH_PREDICTION))
        if fname.endswith(".nc") and fname.startswith("PRED_") 
        and missing_perc in fname
    ]

y_files_trans = [
        xr.open_dataset(os.path.join(DATA_PATH_PREDICTION, fname))['mean_rec'].values
        for fname in sorted(os.listdir(DATA_PATH_PREDICTION))
        if fname.endswith(".nc") and fname.startswith("Y_")  
        and missing_perc in fname
    ]

x_files = [
        xr.open_dataset(os.path.join(DATA_PATH_PREDICTION, fname))['mean_rec'].values
        for fname in sorted(os.listdir(DATA_PATH_PREDICTION))
        if fname.endswith(".nc") and fname.startswith("X_")  
        and missing_perc in fname
    ]

print(np.nanmax(pred_files_trans[0]))
print(np.nanmin(pred_files_trans[0]))
print(np.nanmax(y_files_trans[0]))
print(np.nanmin(y_files_trans[0]))

pred_files = []
std_files = []
y_files = []
x_files_trans = []
for p in range(len(pred_files_trans)):
    if 'multi' in file_names[p]:
        lambda_cb = -0.2816107171659248
        max_val = 194.84649658203125
        min_val = 0.12770438194274902
        
    elif 'olci' in file_names[p]:
        lambda_cb = -0.08346259836955258
        max_val = 171.9263916015625
        min_val = 1.000003457069397
    
    preds = pred_files_trans[p]
    stds = std_files_trans[p]
    ys = y_files_trans[p]
    xs_trans = x_files[p]
    ### When Boxcox transformed
    # inverse normalize
    # preds = preds * (max_val_cb - min_val_cb) + min_val_cb
    # stds= stds * (max_val_cb - min_val_cb) + min_val_cb
    # ys =  ys * (max_val_cb - min_val_cb) + min_val_cb
    
    # divide by 10
    preds = preds/10
    stds = stds/10
    ys = ys/10
    
    # inverse coxbox
    # preds = (preds*lambda_cb + 1.0 )**(1/lambda_cb)
    # stds = (stds*lambda_cb + 1.0 )**(1/lambda_cb)
    # ys = (ys*lambda_cb + 1.0 )**(1/lambda_cb)
    # xs_trans = (xs_trans **(lambda_cb)-1)/(lambda_cb)
    
    # xs_trans = (xs_trans - min_val_cb)/(max_val_cb - min_val_cb)
    
    # When log transformed
    preds = np.exp(preds)
    stds = np.exp(stds)
    ys = np.exp(ys)
    xs_trans = np.log(xs_trans)
    
    # #### When just normalized
    # inverse normalize
    # preds = preds * (max_val - min_val) + min_val
    # stds= stds * (max_val - min_val) + min_val
    # ys =  ys * (max_val - min_val) + min_val
    
    # xs_trans = (xs_trans - min_val)/(max_val - min_val)
    
    xs_trans = xs_trans*10
    
    
    pred_files.append(preds)
    std_files.append(stds)
    y_files.append(ys)
    x_files_trans.append(xs_trans)


print(len(pred_files), len(y_files), len(x_files), len(file_numbers))
assert len(file_numbers) == len(pred_files) == len(y_files) ==len(x_files)

with open(os.path.join(DATA_PATH_PREDICTION,f'Prediction_times.txt'), 'r') as f:
    times = f.readlines()
times = times[0].split(":")[1]
times = [float(t) for t in times.split(", ")[:-1]]
print(len(times))
mses = []
sccs=[]
for pred_file, y_file, x_file, file_number, time_ in zip(pred_files, y_files, x_files, file_numbers, times):
    if np.nansum(y_file) != 0:
        print(file_number)
        mse, scc_value = mse_gaps()(y_file, pred_file, x_file)
        mses.append(mse)
        sccs.append(scc_value)
        with open(os.path.join(DATA_PATH_PREDICTION,'Prediction_validation_multi.txt'), 'a') as f:
                    f.write(f'{file_number}, {time_}, {mse}, {scc_value}, True \n')

print("MSE")            
print(np.nanmean(mses))
print(mses)

print("RMSE")
rmses = np.sqrt(mses)
print(np.nanmean(np.sqrt(mses)))
print(rmses)
print("SCC")
print(np.nanmean(sccs))
print(sccs)

#%%
import scipy
# calculate 95% confidence intervals
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    print(n)
    m, se = np.nanmean(a), scipy.stats.sem(a, nan_policy='omit')
    h = se * scipy.stats.t.ppf((1+confidence)/2., n-1)
    return m, h

print(mean_confidence_interval(mses))   
print(mean_confidence_interval(rmses))


#%%
#print max values for one time series 
value = 10
y_pred = pred_files[value]
y_train = x_files[value]
y_true = y_files[value]

y_pred_trans = pred_files_trans[value]
y_train_trans = x_files_trans[value]
y_true_trans = y_files_trans[value]

mask_ = np.ma.masked_where(~np.isnan(y_train), y_train)
mask = np.ma.getmask(mask_)

y_pred_masked = np.ma.masked_array(y_pred, mask=mask)
y_true_masked = np.ma.masked_array(y_true, mask=mask)

y_pred_masked_trans = np.ma.masked_array(y_pred_trans, mask=mask)
y_true_masked_trans = np.ma.masked_array(y_true_trans, mask=mask)

y_true_val = np.ma.masked_array(y_true_masked,mask=np.isnan(y_true_masked))
y_pred_val = np.ma.masked_array(y_pred_masked,mask=np.isnan(y_true_masked))

y_true_val_trans = np.ma.masked_array(y_true_masked_trans,mask=np.isnan(y_true_masked))
y_pred_val_trans = np.ma.masked_array(y_pred_masked_trans,mask=np.isnan(y_true_masked))

#%%

fig, ax = plt.subplots(2,2, figsize=(7,5))
p1 = ax[0,0].imshow(np.nanmax(y_true_trans, axis=0), origin="lower")
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])
ax[0,0].set_title(f"Observed \n transformed max value = {np.nanmax(y_true_trans):.2f}")
p2 = ax[0,1].imshow(np.nanmax(y_pred_trans, axis=0), origin="lower")
ax[0,1].set_xticks([])
ax[0,1].set_yticks([])
ax[0,1].set_title(f"Predicted \n transformed max value = {np.nanmax(y_pred_trans):.2f}")


p3 = ax[1,0].imshow(np.nanmax(y_true, axis=0), vmin=0, origin="lower")
ax[1,0].set_xticks([])
ax[1,0].set_yticks([])
ax[1,0].set_title(f"Observed \n max value = {np.nanmax(y_true):.2f}")
p4 = ax[1,1].imshow(np.nanmax(y_pred, axis=0),
                  vmin=0, origin="lower")
ax[1,1].set_xticks([])
ax[1,1].set_yticks([])
ax[1,1].set_title(f"Predicted \n max value = {np.nanmax(y_pred):.2f}")

fig.text(-2., 0.55, 'Coxbox scale', horizontalalignment='center',
  verticalalignment='center', transform=ax[0,1].transAxes, fontsize=10, rotation=90)

fig.text(-2., -0.9, 'Linear scale', horizontalalignment='center',
  verticalalignment='center', transform=ax[0,1].transAxes, fontsize=10, rotation=90)

plt.colorbar(p1, ax=ax[0,0], fraction=0.046, pad=0.04)
plt.colorbar(p2, ax=ax[0,1], fraction=0.046, pad=0.04)
plt.colorbar(p3, ax=ax[1,0], fraction=0.046, pad=0.04)
plt.colorbar(p4, ax=ax[1,1], fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()

    
#%%
print(file_numbers)
from mpl_toolkits.axes_grid1 import make_axes_locatable

#plt.tight_layout()

for i in range(8): #len(file_numbers)
    print(file_numbers[i])
    x_file = x_files[i]
    y_file = y_files[i]
    
    pred_file = pred_files[i]
    std_file = std_files[i]
        
    vmin = 0
    vmax = 40
    
    print(np.nanmax(x_file))
    print(np.nanmax(y_file))
    
    print(x_file.shape, y_file.shape)
    f, axes = plt.subplots(5, 5, figsize=(6,7), layout='constrained')
    axes[0,0].set_title("Prediction")
    axes[0,1].set_title("True Data")
    axes[0,2].set_title("Input Data")
    axes[0,3].set_title("Predicted\nUncertainty")
    axes[0,4].set_title("Difference")

    for a in range(0,5):
        axes[a,0].set_ylabel(f"Day {a}")
        
        img = pred_file[a,:,:]
        pic1 = axes[a,0].imshow( img, origin='lower', cmap='viridis',vmin=vmin, vmax=vmax)
        axes[a,0].set_xticks([])
        axes[a,0].set_yticks([])
        
        axes[a,1].imshow(y_file[a,:,:], origin='lower', cmap='viridis',vmin=vmin, vmax=vmax)
        axes[a,1].set_xticks([])
        axes[a,1].set_yticks([])
        axes[a,2].imshow(np.where(np.isnan(x_file[a,:,:]), np.nan, y_file[a,:,:]), origin='lower', cmap='viridis',vmin=vmin, vmax=vmax)
        axes[a,2].set_xticks([])
        axes[a,2].set_yticks([])
        
        pic2 = axes[a,3].imshow(std_file[a,:,:]-1, cmap='YlOrBr',origin='lower', vmin=0, vmax=0.25)
        axes[a,3].set_xticks([])
        axes[a,3].set_yticks([])
        
        pic3 = axes[a,4].imshow(np.where(~np.isnan(y_file[a,:,:]), np.subtract(pred_file[a,:,:],y_file[a,:,:]), np.nan), 
                               origin='lower', cmap='RdYlGn', vmin=-10, vmax=10)
        axes[a,4].set_xticks([])
        axes[a,4].set_yticks([])
        
    f.suptitle(f"Model DINCAE, {missing_perc[1:3]}% missing", fontsize = 20)    
    f.subplots_adjust(bottom=0.1, top=0.9, left=0.02, right=1.1,
                    wspace=0.4, hspace=0.2)
    cbar1 = f.colorbar(pic1, ax=axes[4,:3], shrink=0.8, location='bottom', aspect=25)
    cbar1.set_label('Chlorophyll concentration [mg/m$^{-3}$]')
    cbar2 = f.colorbar(pic2, ax=axes[4,3], shrink=0.8, location='bottom', aspect=7)
    cbar2.set_label('Uncertainty')
    cbar3 = f.colorbar(pic3, ax=axes[4,-1], shrink=0.8, location='bottom', aspect=7)
    cbar3.set_label('Error [mg/m$^{-3}$]')
    #plt.tight_layout()    
    f.savefig(f"{DATA_PATH_PREDICTION}/{i}_visualization_{missing_perc}.png", dpi=300)
    
#plt.show()
print(len(pred_files))

#%%

# plot difference between two days
plt.figure()
plt.imshow(pred_files_trans[0][1,:,:] - pred_files_trans[0][15,:,:], origin='lower')
plt.colorbar()
plt.show()

        
#%%
        
# LOOP OVER THE PREDICTIONS TO GENERATE THE HISTOGRAMS
        
preds = np.empty(1)
ys = np.empty(1)

preds_real = np.empty(1)
ys_real = np.empty(1)

val_preds = np.empty(1)
val_ys = np.empty(1)

val_ys_real = np.empty(1)
val_preds_real = np.empty(1)

for i in range(len(pred_files)):
    p = pred_files_trans[i]
    y = y_files_trans[i]
    preds = np.concatenate((preds, p.flatten()), axis = 0)
    ys = np.concatenate((ys, y.flatten()), axis = 0)
    
    pr = pred_files[i]
    yr = y_files[i]
    preds_real = np.concatenate((preds_real, pr.flatten()), axis = 0)
    ys_real = np.concatenate((ys_real, yr.flatten()), axis = 0)
    
for i in range(len(pred_files)):

    p = pred_files_trans[i]
    y = y_files_trans[i]
    
    p1 = np.where(~np.isnan(y), p, np.nan)
    p1 = np.where(np.isnan(x_files[i]), p1, np.nan)
    
    y1 = np.where(np.isnan(x_files[i]), y, np.nan)
    y1 = np.where(~np.isnan(p1), y1, np.nan)
    
    assert p1[~np.isnan(p1)].shape == y1[~np.isnan(y1)].shape
    
    val_preds = np.concatenate((val_preds, p1.flatten()), axis = 0)
    val_ys = np.concatenate((val_ys, y1.flatten()), axis = 0)
    
    if 'multi' in file_names[i]:
        lambda_cb = -0.2816107171659248
        max_val = 194.84649658203125
        min_val = 0.12770438194274902
        max_val_cb = 2.7464589875132868
        min_val_cb = -2.7884672658092775
        
    elif 'olci' in file_names[i]:
        lambda_cb = -0.08346259836955258
        max_val = 171.9263916015625
        min_val = 1.000003457069397
    
    # y_real = ((y1/10)*(lambda_cb) + 1.0 )**(1/(lambda_cb))
    # p_real = ((p1/10)*(lambda_cb) + 1.0 )**(1/(lambda_cb))
    y_real = np.exp(y1) #/10)
    p_real = np.exp(p1) #/10)
    # y_real = y1/10 * (max_val - min_val) + min_val
    # p_real = p1/10 * (max_val - min_val) + min_val
    
    print(p_real[~np.isnan(p_real)].shape, y_real[~np.isnan(y_real)].shape)
    assert p_real[~np.isnan(p_real)].shape == y_real[~np.isnan(y_real)].shape
    val_ys_real = np.concatenate((val_ys_real, y_real.flatten()), axis=0)
    val_preds_real = np.concatenate((val_preds_real, p_real.flatten()), axis=0)


ys = ys[1:]#[~np.isnan(ys[1:])]
preds = preds[1:]#[~np.isnan(preds[1:])]
val_ys = val_ys[1:]#[~np.isnan(val_ys[1:])]
val_preds = val_preds[1:]#[~np.isnan(val_preds[1:])]
ys_real = ys_real[1:]#[~np.isnan(val_ys[1:])]
preds_real = preds_real[1:]
val_ys_real = val_ys_real[1:]
val_preds_real = val_preds_real[1:]
print("True Values of All Pixels:")
print(np.nanmax(ys))
print(np.nanmin(ys))
print("Predicted Values of All Pixels:")
print(np.nanmax(preds))
print(np.nanmin(preds))

print("Shape of the true pixels")
print(ys.shape)
print(preds.shape)

print("Shape of the val pixels")
print(val_ys_real.shape)
print(val_preds_real.shape)

assert val_ys_real[~np.isnan(val_ys_real)].shape == val_preds_real[~np.isnan(val_preds_real)].shape

print("True Values of Validation Pixels:")
print(np.nanmax(val_ys))
print(np.nanmin(val_ys))
print("Predicted Values of Validation Pixels:")
print(np.nanmax(val_preds))
print(np.nanmin(val_preds))


print("Shape of the val pixels")
print(val_ys.shape)
print(val_preds.shape)

print("True Values of All Pixels True scale:")
print(np.nanmax(ys_real))
print(np.nanmin(ys_real))
print("Predicted Values of All Pixels True scale:")
print(np.nanmax(preds_real))
print(np.nanmin(preds_real))

#%%
# val_ys_trans = (val_ys**(-0.1745758230162425)-1)/(-0.1745758230162425)
# val_ys_std = (val_ys_trans - np.nanmean(val_ys_trans))/np.nanstd(val_ys_trans)

# ys_trans = (ys**(-0.1745758230162425)-1)/(-0.1745758230162425)
# ys_std = (ys_trans - np.nanmean(ys_trans))/np.nanstd(ys_trans)

# val_preds_trans = (val_preds**(-0.1745758230162425)-1)/(-0.1745758230162425)
# preds_trans = (preds**(-0.1745758230162425)-1)/(-0.1745758230162425)

#%%
#------------------------------------------------------------
#------------------------------------------------------------
#### PLOT THE RESIDUALS OF THE PREDICTIONS
#------------------------------------------------------------
#------------------------------------------------------------

def generate_pred_histogram(chl, preds, prefix, path, fig_height=4.5):
        sns.set_theme(style="whitegrid")
        print("Start generating histogram...")
        min_val = -20
        max_val = 40
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
            os.path.join(path, f"{prefix}_histogram.png"), bbox_inches="tight", pad_inches=0.01
        )
        plt.show()
        
#%%
        
#generate_pred_histogram(val_ys, val_preds, f"chl_{model}_log_zscaled_validationpixels", DATA_PATH_PREDICTION) # 
generate_pred_histogram(val_ys, val_preds, f"chl_{model_name}_val", DATA_PATH_PREDICTION) # 

#%%
print("Plotting the Residuals")
fig, ax = plt.subplots(3,2, figsize=(10,10))
ax[0,0].hist(val_ys, bins=100)
ax[0,0].set_title("True Distribution of Validation Pixels")
#ax[0,0].set_yscale('log')

#val_ys_real = np.exp(np.exp(val_ys/10))
#val_preds_real = np.exp(np.exp(val_preds/10))

pred_graph = val_preds[~np.isnan(val_preds)]
ax[0,1].hist(np.where(np.isfinite(val_preds) == True, val_preds, np.nan), bins=100)
ax[0,1].set_title("Predicted Distribution of Validation Pixels")
#ax[0,1].set_yscale('log')

ax[1,0].scatter(val_ys, np.subtract(val_preds,val_ys), alpha=0.2)
ax[1,0].set_ylabel("Residuals")
ax[1,1].scatter(val_ys_real, np.subtract(val_preds_real, val_ys_real), alpha=0.2)
ax[1,1].set_ylabel("Residuals")

ax[2,0].scatter(ys, np.subtract(preds,ys), alpha=0.2)
ax[2,0].set_ylabel("Residuals")
ax[2,1].scatter(ys_real, np.subtract(preds_real, ys_real), alpha=0.2)
ax[2,1].set_ylabel("Residuals")
plt.tight_layout()
plt.show()

#%%
# determining the R2 value
from sklearn.metrics import r2_score
print("R2 Score of the Validation Pixels:")
print(r2_score(val_ys[~np.isnan(val_ys)], val_preds[~np.isnan(val_preds)]))

print("R2 Score of the Validation Pixels True scale:")
print(r2_score(val_ys_real[~np.isnan(val_ys_real)], val_preds_real[~np.isnan(val_preds_real)]))

print("R2 Score of the Validation Pixels Log True scale:")
print(r2_score(np.log(val_ys_real[~np.isnan(val_ys_real)]), np.log(val_preds_real[~np.isnan(val_preds_real)])))


#%%
ys_filt_ = ys[~np.isnan(preds)]
ys_filt = ys_filt_[~np.isnan(ys_filt_)]
preds_filt_ = preds[~np.isnan(ys)]
preds_filt = preds_filt_[~np.isnan(preds_filt_)]

ys_r_filt_ = ys_real[~np.isnan(preds_real)]
ys_r_filt = ys_r_filt_[~np.isnan(ys_r_filt_)]
preds_r_filt_ = preds_real[~np.isnan(ys_real)]
preds_r_filt = preds_r_filt_[~np.isnan(preds_r_filt_)]
from sklearn.metrics import r2_score
print("R2 Score of the Transformed Pixels:")
print(r2_score(ys_filt, preds_filt))

print("R2 Score of the True Scale pixels:")
print(r2_score(ys_r_filt[~np.isnan(ys_r_filt)], preds_r_filt[~np.isnan(preds_r_filt)]))


# %%
