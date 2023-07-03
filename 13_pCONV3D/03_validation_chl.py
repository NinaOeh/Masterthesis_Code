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
from matplotlib.colors import ListedColormap


model_name = "model_213_KTT_dataperc30-100_3"

DATA_PATH_PREDICTION  = f"./training_results/prediction_{model_name}"


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
    
    y_true_new = y_true[~np.isnan(y_pred)]
    y_pred_new = y_pred[~np.isnan(y_pred)]
    
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

def mse_gaps(mask, validation_mask, land_mask):
    """
    Computes MSE for gap values only
    """
    def loss(y_true, y_pred):
        # mask value 0 -> NO DATA, mask value 1 -> DATA
        # landmask value 0 -> OCEAN, mask value 1 -> LAND
        if model_name != "DINEOF":
            y_pred = y_pred.swapaxes(1, 2)
            y_pred = y_pred.swapaxes(0, 1)
            y_true = y_true.swapaxes(1, 2)
            y_true = y_true.swapaxes(0, 1)
        
        # val_mask_masked = np.where(land_mask==0, validation_mask , 0)
        # val_mask_masked = np.where(mask==1, val_mask_masked , 0)
        val_mask_masked = np.where(mask==1, validation_mask , 0)
        val_mask_masked = np.where(val_mask_masked!=0, val_mask_masked , 0)
        val_mask_indices = np.where(val_mask_masked==1)
        y_true = np.nan_to_num(y_true)
        y_pred = np.nan_to_num(y_pred)
        
        y_true_oneD = y_true[val_mask_indices[2], val_mask_indices[1], val_mask_indices[0]]
        
        if model_name == "DINEOF":
            y_pred = np.exp(y_pred) - 1
            y_pred_oneD = y_pred[val_mask_indices[2], val_mask_indices[1], val_mask_indices[0]]
        else:
            
            y_pred_oneD = y_pred[val_mask_indices[2], val_mask_indices[1], val_mask_indices[0]]
        
        #mse = mean_squared_error(y_true, y_pred)
        mse = mean_squared_error(y_true_oneD, y_pred_oneD)
        scc_value = spatial_correlation_coefficient(y_true, y_pred)
        return mse, scc_value
    return loss

missing_perc = "_" #"_20_"

pred_file_names = [
            os.path.basename(os.path.join(DATA_PATH_PREDICTION, fname))
            for fname in sorted(os.listdir(DATA_PATH_PREDICTION))
            if fname.endswith(".tiff") and fname.startswith("PRED_") 
            and missing_perc  in fname
        ]

pred_file_numbers = [file_name.split("_")[-1].split(".")[0] for file_name in pred_file_names]

y_file_names = [
            os.path.basename(os.path.join(DATA_PATH_PREDICTION, fname)).split("_")[-1]
            for fname in sorted(os.listdir(DATA_PATH_PREDICTION))
            if fname.endswith(".tiff") and fname.startswith("Y_")  
            and missing_perc  in fname
        ]

y_file_numbers = [file_name.split(".")[0] for file_name in y_file_names]


pred_files_trans = [
        skio.imread(os.path.join(DATA_PATH_PREDICTION, fname))
        for fname in sorted(os.listdir(DATA_PATH_PREDICTION))
        if fname.endswith(".tiff") and fname.startswith("PRED_") 
        and missing_perc  in fname
    ]
print("Min and max of Predictions transformed")
print(np.nanmax(pred_files_trans))
print(np.nanmin(pred_files_trans))

x_files = [
        skio.imread(os.path.join(DATA_PATH_PREDICTION, fname))
        for fname in sorted(os.listdir(DATA_PATH_PREDICTION))
        if fname.endswith(".tiff") and fname.startswith("X_") 
        and missing_perc  in fname
    ]

y_files_trans = [
        skio.imread(os.path.join(DATA_PATH_PREDICTION, fname))
        for fname in sorted(os.listdir(DATA_PATH_PREDICTION))
        if fname.endswith(".tiff") and fname.startswith("Y_")  #and 'olci' not in fname
        and missing_perc  in fname
    ]
print("Min and max of Y files transformed")
print(np.nanmax(y_files_trans))
print(np.nanmin(y_files_trans))

valmask_files = [
            skio.imread(os.path.join(DATA_PATH_PREDICTION, fname))
            for fname in sorted(os.listdir(DATA_PATH_PREDICTION))
            if fname.endswith(".tiff") and fname.startswith("VALMASK_") #and 'olci' not in fname
            and missing_perc  in fname
        ]
mask_files = [
            skio.imread(os.path.join(DATA_PATH_PREDICTION, fname))
            for fname in sorted(os.listdir(DATA_PATH_PREDICTION))
            if fname.endswith(".tiff") and fname.startswith("MASK_") #and 'olci' not in fname
            and missing_perc  in fname
        ]
landmask_files = [
            skio.imread(os.path.join(DATA_PATH_PREDICTION, fname))
            for fname in sorted(os.listdir(DATA_PATH_PREDICTION))
            if fname.endswith(".tiff") and fname.startswith("LANDMASK_") #and 'olci' not in fname
            and missing_perc  in fname
        ]

assert len(pred_files_trans) == len(y_files_trans) == len(valmask_files) == len(landmask_files) == len(mask_files)

print(pred_file_numbers)
print(y_file_numbers)

pred_files = []
y_files = []
x_files_trans = []
for p in range(len(pred_file_numbers)):
    print(pred_file_names[p])
    # if 'multi' in pred_file_names[p]:
    #     lambda_cb = -0.015201357272472429
    #     max_val = 194.71963500976562
    #     min_val = 0.1258625090122223
        
    # elif 'olci' in pred_file_names[p]:
    #     lambda_cb = -0.306205311552774
    #     max_val = 154.74884033203125
    #     min_val = 0.21534013748168945
        
    if 'multi' in pred_file_names[p]:
        lambda_cb = -0.2816107171659248
        max_val = 194.84649658203125
        min_val = 0.12770438194274902
    elif 'olci' in pred_file_names[p]:
        lambda_cb = -0.08346259836955258
        max_val = 171.9263916015625
        min_val = 1.000003457069397
    
    preds = pred_files_trans[p]
    ys = y_files_trans[p]
    xs_trans = x_files[p]
    ### When Boxcox transformed
    # inverse normalize
    # preds = preds * (max_val_cb - min_val_cb) + min_val_cb
    # stds= stds * (max_val_cb - min_val_cb) + min_val_cb
    # ys =  ys * (max_val_cb - min_val_cb) + min_val_cb
    
    # # divide by 10
    # preds = preds/10
    # ys = ys/10
    
    # inverse coxbox
    # preds = (preds*lambda_cb + 1.0 )**(1/lambda_cb)
    # ys = (ys*lambda_cb + 1.0 )**(1/lambda_cb)
    # xs_trans = (xs_trans **(lambda_cb)-1)/(lambda_cb)
    
    #xs_trans = (xs_trans - min_val_cb)/(max_val_cb - min_val_cb)
    
    # When log transformed
    preds = np.exp(preds)
    ys = np.exp(ys)
    xs_trans = np.log(xs_trans)
    
    # #### When just normalized
    # inverse normalize
    # preds = preds * (max_val - min_val) + min_val
    # ys =  ys * (max_val - min_val) + min_val
    
    # xs_trans = (xs_trans - min_val)/(max_val - min_val)
    
    
    pred_files.append(preds)
    y_files.append(ys)
    x_files_trans.append(xs_trans)
        

with open(os.path.join(DATA_PATH_PREDICTION,f'Prediction_times.txt'), 'r') as f:
    times = f.readlines()
times = times[0].split(":")[1]
times = [float(t) for t in times.split(", ")[:-1]]
print(len(times))
mses = []
sccs=[]

for i in range(len(pred_file_numbers)):
    pred_name = pred_file_names[i]
    
    pred_file = pred_files[i]
    y_file = y_files[i]
    valmask_file = valmask_files[i]
    landmask_file = landmask_files[i]
    mask_file = mask_files[i]
    file_number = pred_file_numbers[i]
    #time_ = times[i]

    print(f"File name: {pred_name}")
    mse, scc_value = mse_gaps(mask_file, valmask_file, landmask_file)(y_file, pred_file)
    mses.append(mse)
    sccs.append(scc_value)
    # with open(os.path.join(DATA_PATH_PREDICTION,'Prediction_validation_olci.txt'), 'a') as f:
    #             f.write(f'{file_number}, {time_}, {mse}, {scc_value}, True \n')

print("MSE")            
#print(np.nanmean(mses))
#mses = np.concatenate(mses)
print(mses)
#print(mses.shape)
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
file_name = 155
print(pred_file_numbers)
pred_file_number = pred_file_numbers.index(str(file_name))
vmax=40
print(pred_file_numbers)
value = 10
f, a = plt.subplots(1,3, figsize=(9,5))
y_masked = np.where(mask_files[pred_file_number] == 1, y_files[pred_file_number], np.nan)[:,:,value]
p0 = a[0].imshow(np.where(landmask_files[pred_file_number][:,:,value] == 1, y_masked , np.nan), origin="lower", 
                 vmin=0, vmax=vmax, cmap='viridis')
a[0].set_title("True")
a[0].set_xticks([])
a[0].set_yticks([])
p1 = a[1].imshow(np.where(landmask_files[pred_file_number] == 1, pred_files[pred_file_number], 
                          np.nan)[:,:,value], origin="lower", vmin=0, vmax=vmax, cmap='viridis')
a[1].set_title("Gap Filled")
a[1].set_xticks([])
a[1].set_yticks([])
p2 = a[2].imshow(np.where(landmask_files[pred_file_number] == 1, pred_files[pred_file_number] - y_files[pred_file_number], 
                          np.nan)[:,:,value], origin="lower", cmap='RdYlGn',
                          vmin=-2, vmax=2)
a[2].set_title("Error")
a[2].set_xticks([])
a[2].set_yticks([])

cbar1 = f.colorbar(p0, ax=a[:2], shrink=0.8, location='bottom', aspect=25, pad=0.1)
cbar1.set_label('Chlorophyll concentration [mg/m$^{3}$]')
cbar3 = f.colorbar(p2, ax=a[-1], shrink=0.8, location='bottom', aspect=12, pad=0.1)
cbar3.set_label('Error [mg/m$^{3}$]')
#plt.tight_layout()
plt.show()


print(np.nanmax(y_files[pred_file_number]))
print(np.nanmax(pred_files[pred_file_number]))

#%%

y_pred = pred_files[pred_file_number]
    
y_file = y_files[pred_file_number]
landmask_file = landmask_files[pred_file_number]
fig, ax = plt.subplots(2,2, figsize=(7,5))
p1 = ax[0,0].imshow(np.nanmax(np.log(y_file), axis=2), vmin=0, vmax=4, origin="lower", cmap='viridis')
ax[0,0].set_xticks([])
ax[0,0].set_yticks([])
ax[0,0].set_title(f"Observed \n log max value = {np.nanmax(np.log(y_file)):.2f}")
p2 = ax[0,1].imshow(np.nanmax(np.where(landmask_file == 1, np.log(y_pred), np.nan), axis=2), 
                    cmap='viridis', vmin=0, vmax=4, origin="lower")
ax[0,1].set_xticks([])
ax[0,1].set_yticks([])
ax[0,1].set_title(f"Predicted \n log max value = {np.nanmax(np.log(y_pred)):.2f}")


p3 = ax[1,0].imshow(np.where(landmask_file[:,:,0] == 1,np.nanmax(y_file, axis=2),np.nan), 
                    vmin=0, vmax=30, origin="lower", cmap='viridis')
ax[1,0].set_xticks([])
ax[1,0].set_yticks([])
ax[1,0].set_title(f"Observed \n max value = {np.nanmax(y_file):.2f}")
p4 = ax[1,1].imshow(np.nanmax(np.where(landmask_file == 1, y_pred, np.nan), axis=2),
                  vmin=0, vmax=30,  origin="lower" , cmap='viridis' )
ax[1,1].set_xticks([])
ax[1,1].set_yticks([])
ax[1,1].set_title(f"Predicted \n max value = {np.nanmax(y_pred):.2f}")

fig.text(-2., 0.55, 'Logarithmic scale', horizontalalignment='center',
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
print("Plotting an animation")
y_pred = pred_files[0].swapaxes(1, 2)
y_pred = y_pred.swapaxes(0, 1)

y_file = y_files[0].swapaxes(1, 2)
y_file = y_file.swapaxes(0, 1)

landmask_file = landmask_files[0].swapaxes(1, 2)
landmask_file = landmask_file.swapaxes(0, 1)

valmask_file = valmask_files[0].swapaxes(1, 2)
valmask_file = valmask_file.swapaxes(0, 1)

mask_file = mask_files[0].swapaxes(1, 2)
mask_file = mask_file.swapaxes(0, 1)

fig = plt.figure()
images = [
    [plt.imshow(
        layer, animated=True, origin="lower"
    )]
    for layer in np.where(landmask_file==1, y_pred, np.nan)
]
animation_3d = animation.ArtistAnimation(fig, images, interval=20, blit=True)
plt.show()

#%%
filled_file = np.where(valmask_file==1, y_pred, y_file)
filled_file = np.where(landmask_file==1, filled_file, np.nan)
filled_file = np.where(mask_file==0, y_pred, filled_file)

fig = plt.figure()
images = [
    [plt.imshow(
        layer, animated=True, origin="lower"
    )]
    for layer in filled_file
]
animation_3d = animation.ArtistAnimation(fig, images, interval=20, blit=True)
plt.show()

#%%
fig = plt.figure()
images = [
    [plt.imshow(
        layer, animated=True, origin="lower"
    )]
    for layer in y_file
]
animation_3d = animation.ArtistAnimation(fig, images, interval=20, blit=True)
plt.show()
    
    
#%%

for i in range(25,28): 
    x_file = x_files[i]
    y_file = y_files[i]
    
    pred_file = pred_files[i]
        
    vmin = 0
    vmax = 10
    
    print(np.nanmax(x_file))
    print(np.nanmax(y_file))
    
    print(x_file.shape, y_file.shape)
    f, axes = plt.subplots(5, 4, figsize=(6,7), layout='constrained')
    axes[0,0].set_title("Prediction")
    axes[0,1].set_title("True Data")
    axes[0,2].set_title("Input Data")
    axes[0,3].set_title("Difference")

    for a in range(0,5):
        axes[a,0].set_ylabel(f"Day {a}")
        
        img = np.where(landmask_file[:,:,a]==1, pred_file[:,:,a], np.nan)
        pic1 = axes[a,0].imshow( img, origin='lower', cmap='viridis',vmin=vmin, vmax=vmax)
        axes[a,0].set_xticks([])
        axes[a,0].set_yticks([])
        
        axes[a,1].imshow(y_file[:,:,a], origin='lower', cmap='viridis',vmin=vmin, vmax=vmax)
        axes[a,1].set_xticks([])
        axes[a,1].set_yticks([])
        axes[a,2].imshow(np.where(np.isnan(x_file[:,:,a]), np.nan, y_file[:,:,a]), origin='lower', cmap='viridis',vmin=vmin, vmax=vmax)
        axes[a,2].set_xticks([])
        axes[a,2].set_yticks([])
        
        pic3 = axes[a,3].imshow(np.where(~np.isnan(y_file[:,:,a]), np.subtract(pred_file[:,:,a],y_file[:,:,a]), np.nan), 
                               origin='lower', cmap='RdYlGn', vmin=-10, vmax=10)
        axes[a,3].set_xticks([])
        axes[a,3].set_yticks([])
        
    f.suptitle(f"Model pCONV3D, Gotland", fontsize = 20)    
    f.subplots_adjust(bottom=0.1, top=0.9, left=0.02, right=1.1,
                    wspace=0.4, hspace=0.2)
    cbar1 = f.colorbar(pic1, ax=axes[4,:3], shrink=0.8, location='bottom', aspect=25)
    cbar1.set_label('Chlorophyll concentration [mg/m$^{3}$]')
    cbar3 = f.colorbar(pic3, ax=axes[4,-1], shrink=0.8, location='bottom', aspect=7)
    cbar3.set_label('Error [mg/m$^{3}$]')
    #plt.tight_layout()    
    f.savefig(f"{DATA_PATH_PREDICTION}/{i}_visualization_{missing_perc}.png", dpi=300)
    
#plt.show()
print(len(pred_files))

#%%
#------------------------------------------------------------
        
# LOOP OVER THE PREDICTIONS TO GENERATE THE HISTOGRAMS
        
preds = []
ys = []

val_preds = []
val_ys = []

val_ys_real = []
val_preds_real = []

ys_real = []
preds_real = []

plt.figure()
plt.imshow(landmask_files[0][:,:,0])
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(mask_files[0][:,:,0])
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(valmask_files[0][:,:,0])
plt.colorbar()
plt.show()

for i in range(len(pred_files)):
    p = pred_files_trans[i]
    mask = mask_files[i]
    valmask = valmask_files[i]
    y = y_files_trans[i]

    p = np.where(landmask_files[0] == 1, p, np.nan)
    p = np.where(mask != 0, p, np.nan)
    preds.append(p.flatten())
    
    y = np.where(landmask_files[0] == 1, y, np.nan)
    y = np.where(mask != 0, y, np.nan)
    y = np.where(y == 0, np.nan, y)
    ys.append(y.flatten())
    
    ys_real.append(np.exp(y.flatten()))
    preds_real.append(np.exp(p.flatten()))
    
    p_ = p.copy()
    p_ = np.where(valmask != 0, p_, np.nan)
    val_preds.append(p_.flatten())
    
    y_ = y.copy()
    y_ = np.where(valmask != 0, y_, np.nan)
    val_ys.append(y_.flatten())

    
    if 'multi' in pred_file_names[i]:
        lambda_cb = -0.2816107171659248
        max_val = 194.84649658203125
        min_val = 0.12770438194274902
        max_val_cb = 2.7464589875132868
        min_val_cb = -2.7884672658092775
        
    elif 'olci' in pred_file_names[i]:
        lambda_cb = -0.08346259836955258
        max_val = 171.9263916015625
        min_val = 1.000003457069397
        
    # p_real = p_ * (max_val - min_val) + min_val
    # y_real = y_ * (max_val - min_val) + min_val
    
    p_real = np.exp(p_)
    y_real = np.exp(y_)
    
    val_ys_real.append(y_real.flatten())
    val_preds_real.append(p_real.flatten())
 
ys = np.concatenate(ys)
preds = np.concatenate(preds)
print(ys.shape)
print(preds.shape)

print(np.nanmax(val_ys))
print(np.nanmin(val_ys))
print(np.nanmax(val_preds))
print(np.nanmin(val_preds))
print(np.nanmax(val_preds_real))
print(np.nanmin(val_ys_real))

val_ys = np.concatenate(val_ys)
val_preds = np.concatenate(val_preds)

val_ys_real = np.concatenate(val_ys_real)
val_preds_real = np.concatenate(val_preds_real)

ys_real = np.concatenate(ys_real)
preds_real = np.concatenate(preds_real)

print("Shape of the true pixels")
print(ys.shape)
print(preds.shape)

print("Shape of the val pixels")
print(val_ys_real.shape)
print(val_preds_real.shape)

print("Shape of the val pixels")
print(ys_real.shape)
print(preds_real.shape)

#assert val_ys_real[~np.isnan(val_ys_real)].shape == val_preds_real[~np.isnan(val_preds_real)].shape

print("True Values of Validation Pixels:")
print(np.nanmax(val_ys))
print(np.nanmin(val_ys))
print("Predicted Values of Validation Pixels:")
print(np.nanmax(val_preds))
print(np.nanmin(val_preds))

print("True Values of Validation Pixels Real:")
print(np.nanmax(val_ys_real))
print(np.nanmin(val_ys_real))
print("Predicted Values of Validation Pixels Real:")
print(np.nanmax(val_preds_real))
print(np.nanmin(val_preds_real))
print(np.nanmean(preds_real))


print("Shape of the val pixels")
print(val_ys.shape)
print(val_preds.shape)

#%%
#------------------------------------------------------------
#### PLOT THE RESIDUALS OF THE PREDICTIONS
#------------------------------------------------------------
#------------------------------------------------------------

def generate_pred_histogram(chl, preds, prefix, path, fig_height=4.5):
        sns.set_theme(style="whitegrid")
        print("Start generating histogram...")
        min_val = 0
        max_val = 80
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
        plt.savefig(
            os.path.join(path, f"{prefix}_histogram.png"), bbox_inches="tight", pad_inches=0.01
        )
        plt.show()


#%%
        
#generate_pred_histogram(val_ys, val_preds, f"chl_{model}_log_zscaled_validationpixels", DATA_PATH_PREDICTION) # 
generate_pred_histogram(val_ys_real, val_preds_real, f"chl_{model_name}_val_real", DATA_PATH_PREDICTION) # 

#%%

print(np.count_nonzero(~np.isnan(val_ys)))
print(np.count_nonzero(~np.isnan(val_preds)))
print(np.count_nonzero(~np.isnan(ys)))
print(np.count_nonzero(~np.isnan(preds)))

#%%
print("Plotting the Residuals")
fig, ax = plt.subplots(2,2, figsize=(10,5))

# Define colormap
cm = plt.cm.get_cmap('viridis')
my_cmap = cm(np.linspace(0,1,10))

# Create new colormap
my_cmap = ListedColormap(my_cmap)

print(np.shape(val_preds[~np.isnan(val_preds)]))
print(np.shape(val_ys[~np.isnan(val_ys)]))

ax[0,0].grid()
ax[1,0].grid()
ax[0,1].grid()
ax[1,1].grid()

pred_graph = val_preds[~np.isnan(val_preds)]

sns.kdeplot(data=val_ys[~np.isnan(val_ys)], ax=ax[0, 0], label="True", color=my_cmap(1), fill=True)
sns.kdeplot(data=pred_graph, ax=ax[0, 0], label="Predicted", color='green', fill=True)

#ax[0,0].set_yscale('log')
ax[0,0].legend(fontsize=13)
ax[0,0].set_xlabel("Log of Chl-a concentration [log(mg/L)]", fontsize=13)
ax[0,0].set_ylabel("Density", fontsize=13)
ax[0,0].set_title("Log transformed", fontsize=15)

sns.kdeplot(data=val_ys_real[~np.isnan(val_ys_real)], ax=ax[0, 1], label="True", color=my_cmap(1), fill=True)
sns.kdeplot(data=val_preds_real[~np.isnan(val_preds_real)], ax=ax[0, 1], label="Predicted", color='green', fill=True)

#ax[0,1].set_yscale('log')
ax[0,1].legend(fontsize=13)
ax[0,1].set_xlabel("Chl-a concentration [mg/L]", fontsize=13)
ax[0,1].set_ylabel("Density", fontsize=13)
ax[0,1].set_title("True scale", fontsize=15)

ax[1,0].scatter(val_ys, np.subtract(val_preds,val_ys), alpha=0.2, color=my_cmap(4))
ax[1,0].set_ylabel("Residuals", fontsize=13)
ax[1,0].set_xlabel("Observed", fontsize=13)
#ax[1,0].set_title("Log transformed", fontsize=15)
ax[1,1].scatter(val_ys_real, np.subtract(val_preds_real, val_ys_real), alpha=0.2, 
                color=my_cmap(4))
ax[1,1].set_ylabel("Residuals", fontsize=13)
ax[1,1].set_xlabel("Observed", fontsize=13)
#ax[1,1].set_title("True scale", fontsize=15) #"Residuals vs observation \nof validation pixels \n 
#fig.suptitle("Gotland Basin", fontsize=20)
plt.tight_layout()
plt.show()

#%%
fig, axes = plt.subplots(2, 1, figsize=(4, 6))
d = 3
vmin = 0
vmax=40
axes[0].set_ylabel("Occurence [#]", fontsize=13) 
axes[1].set_ylabel("Residuals", fontsize=13)

#axes[0].set_title("Gulf of Riga",  fontsize=20)
# axes[0,3].set_title("(p)")
# axes[0,4].set_title("(q)")
# axes[0,5].set_title("(r)")

axes[0].grid()
axes[1].grid()

pred_graph_ = val_preds_real[~np.isnan(val_preds_real)]
ys_graph_ = val_ys_real[~np.isnan(val_preds_real)]
pred_graph = pred_graph_[~np.isnan(ys_graph_)]
ys_graph = ys_graph_[~np.isnan(ys_graph_)]

sns.kdeplot(data=ys_graph, ax=ax[0, 1], label="True", color=my_cmap(1), fill=True)
sns.kdeplot(data=pred_graph, ax=ax[0, 1], label="Predicted", color='green', fill=True)

axes[0].set_xlabel("Chl-a concentration \n [mg/m$^3$]", fontsize=13)
axes[0].legend(fontsize=11)

axes[1].scatter(val_ys_real, np.subtract(val_preds_real, val_ys_real), alpha=0.2, 
            color=my_cmap(4))
axes[1].set_xlabel("Observed", fontsize=13)
#fig.suptitle("Gotland Basin", fontsize=20)

plt.tight_layout()


#%%
def R_squared(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0,1]
    R_sq = corr**2
    return R_sq
# determining the R2 value for validation pixels
ys_filt_ = val_ys[~np.isnan(val_preds)]
ys_filt = ys_filt_[~np.isnan(ys_filt_)]
preds_filt_ = preds[~np.isnan(val_ys)]
preds_filt = preds_filt_[~np.isnan(preds_filt_)]

ys_r_filt_ = val_ys_real[~np.isnan(val_preds_real)]
ys_r_filt = ys_r_filt_[~np.isnan(ys_r_filt_)]
preds_r_filt_ = val_preds_real[~np.isnan(val_ys_real)]
preds_r_filt = preds_r_filt_[~np.isnan(preds_r_filt_)]

print(len(ys_filt), len(preds_filt))

from sklearn.metrics import r2_score
print("R2 Score of the Transformed Pixels:")
print(r2_score(ys_filt, preds_filt),
      R_squared(ys_filt, preds_filt))

print("R2 Score of the True Scale pixels:")
print(r2_score(ys_r_filt[~np.isnan(ys_r_filt)], preds_r_filt[~np.isnan(preds_r_filt)]),
      R_squared(ys_r_filt[~np.isnan(ys_r_filt)], preds_r_filt[~np.isnan(preds_r_filt)]))

#%%
# determining the R2 value for whole images
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
print(r2_score(ys_filt, preds_filt),
      R_squared(ys_filt, preds_filt))

print("R2 Score of the True Scale pixels:")
print(r2_score(ys_r_filt[~np.isnan(ys_r_filt)], preds_r_filt[~np.isnan(preds_r_filt)]),
      R_squared(ys_r_filt[~np.isnan(ys_r_filt)], preds_r_filt[~np.isnan(preds_r_filt)]))

# %%





# %%
