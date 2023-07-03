#%%
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from tqdm import tqdm
import os
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy import stats  
import scipy

#%%%%%%-----------------------GOTLAND-----------------------%%%%%%%%%%%%%%%

# open the insitu measurements saved in a grid
insitu_file_1 = r"C:\Users\nioh\Documents\MasterThesis\data\InSitu\gtl_ferrybox_measurements_2062023_to_2562023.npy"
insitu_file_2 = r"C:\Users\nioh\Documents\MasterThesis\data\InSitu\gtl_ferrybox_measurements_2662023_to_2862023.npy"
insitu_1 = np.load(insitu_file_1)
insitu_2 = np.load(insitu_file_2)

print(insitu_1.shape)
print(insitu_2.shape)

insitu_total = np.concatenate((insitu_1, insitu_2), axis=0)
print(insitu_total.shape)

#%%
# open the numerical data output

num_dir = r"C:\Users\nioh\Documents\MasterThesis\data\InSitu\CHL_Numerical"
lat_lon_dir = r"C:\Users\nioh\Documents\MasterThesis\data\Numerical_Model_Data\CHL_data_21122021_to_1012022.nc"
df_lat_lon = xr.open_dataset(lat_lon_dir)
chl_data_init = df_lat_lon.sel(time=~df_lat_lon.get_index("time").duplicated())
chl_data_init = chl_data_init.rename_dims({'lon': 'x', 'lat': 'y'}).rename_vars({"lon": "x", "lat": "y"}).set_coords(["x", "y"])
lat_lon_subset = chl_data_init[['x', 'y']]

chl_num_GTL_total = np.load(f"{num_dir}/num_forecast_GTL.npy")
chl_num_RIGA_total = np.load(f"{num_dir}/num_forecast_RIGA.npy")
chl_num_KTT_total = np.load(f"{num_dir}/num_forecast_KTT.npy")

#%%
# open the deep learning model output
deep_dir = r"C:\Users\nioh\Documents\MasterThesis\data\InSitu\CHL_Satellite"

chl_dl_GTL_total = np.load(f"{deep_dir}/deep_learning_preds_GTL.npy")
print(chl_dl_GTL_total.shape)


# %%
insitu_dict = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: []
}

num_dict = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: []
}

dl_dict = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: []
}

rmses_num_in_dict = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: []
}

rmses_dl_in_dict = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: []
}

rmses_dl_num_dict = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: []
}

r2s_num_in_dict = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: []
}

r2s_dl_in_dict = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: []
}

r2s_dl_num_dict = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: []
}


def R_squared(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0,1]
    R_sq = corr**2
    return R_sq

for i in range(5):
    print("EXAMPLE DAY: ", i)
    # adjust the dates to the numerical model output predictions
    insitu = insitu_total[i:i+5,:,:]
    chl_num_GTL = chl_num_GTL_total[i,:,:,:]
    chl_dl_GTL = chl_dl_GTL_total[i,:,:,:]

    chl_num_GTL_filtered = np.where(np.isnan(insitu), np.nan, chl_num_GTL)
    chl_num_GTL_filtered = chl_num_GTL_filtered[~np.isnan(chl_num_GTL_filtered)]
    chl_dl_GTL_filtered = np.where(np.isnan(insitu), np.nan, chl_dl_GTL)
    chl_dl_GTL_filtered = chl_dl_GTL_filtered[~np.isnan(chl_dl_GTL_filtered)]
    insitu_filtered = insitu[~np.isnan(insitu)]

    # compare the insitu measurements with the numerical model output
    # for each day
    rmses_in_num = []
    rmses_in_dl = []
    rmses_dl_num = []
    r2s_in_num = []
    r2s_in_dl = []
    r2s_dl_num = []

    for t in range(5):
        print(f"Day: {t}")
        insit = insitu[t,:,:]
        num = chl_num_GTL[t,:,:]
        dl = chl_dl_GTL[t,:,:]
        
        #print("NUMERICAL - INSITU")
        abs_diff = np.abs(chl_num_GTL[t,:,:] - insitu[t,:,:])
        rmse = np.sqrt(np.nanmean(abs_diff**2))
        
        r2 = R_squared(insit[~np.isnan(insit)], num[~np.isnan(insit)])
        # print("RMSE: ")
        # print(rmse)
        # print("R2: ")
        # print(r2)
        rmses_num_in_dict[t].append(rmse)
        r2s_num_in_dict[t].append(r2)
        insitu_dict[t].append(insit[~np.isnan(insit)])
        num_dict[t].append(num[~np.isnan(insit)])
        rmses_in_num.append(rmse)
        r2s_in_num.append(r2)
        
        print("DL - INSITU")
        abs_diff = np.abs(chl_dl_GTL[t,:,:] - insitu[t,:,:])
        rmse = np.sqrt(np.nanmean(abs_diff**2))
        
        r2 = R_squared(insit[~np.isnan(insit)], dl[~np.isnan(insit)])
        print("RMSE: ")
        print(rmse)
        print("R2: ")
        print(r2)
        rmses_dl_in_dict[t].append(rmse)
        r2s_dl_in_dict[t].append(r2)
        dl_dict[t].append(dl[~np.isnan(insit)])
        rmses_in_dl.append(rmse)
        r2s_in_dl.append(r2)
        
        print("NUMERICAL - DL")
        abs_diff = np.abs(chl_dl_GTL[t,:,:] - chl_num_GTL[t,:,:])
        rmse = np.sqrt(np.nanmean(abs_diff**2))
        
        r2 = R_squared(dl, num)
        print("RMSE: ")
        print(rmse)
        print("R2: ")
        print(r2)
        rmses_dl_num_dict[t].append(rmse)
        r2s_dl_num_dict[t].append(r2)
        rmses_dl_num.append(rmse)
        r2s_dl_num.append(r2)
    
    # print("NUMERICAL - INSITU")  
    # abs_diff = np.abs(chl_num_GTL - insitu)
    # print("RMSE: ")
    # print(np.sqrt(np.nanmean(abs_diff**2)))
    # print("R2: ")
    # print(np.nanmean(r2s_in_num))
    
    print("DL - INSITU")  
    abs_diff = np.abs(chl_dl_GTL - insitu)
    print("RMSE: ")
    print(np.sqrt(np.nanmean(abs_diff**2)))
    print("R2: ")
    print(np.nanmean(r2s_in_dl))
    
    print("DL - NUMERICAL")  
    abs_diff = np.abs(chl_dl_GTL - chl_num_GTL)
    print("RMSE: ")
    print(np.sqrt(np.nanmean(abs_diff**2)))
    print("R2: ")
    print(np.nanmean(r2s_dl_num))
# %%
chl_num_GTL_filtered = np.where(np.isnan(insitu), np.nan, chl_num_GTL)
chl_num_GTL_filtered = chl_num_GTL_filtered[~np.isnan(chl_num_GTL_filtered)]
print(chl_num_GTL_filtered)
insitu_filtered = insitu[~np.isnan(insitu)]
print(insitu_filtered)

#%%
def generate_pred_histogram(chl, preds, prefix, path, d, fig_height=4):
        sns.set_theme(style="whitegrid")
        print("Start generating histogram...")
        min_val = np.min([np.min(chl), np.min(preds)])-0.3
        max_val = np.max([np.max(chl), np.max(preds)])+0.3
        data = {"In-situ chl-a concentration [mg/L]": chl, "Forecast chl-a concentration [mg/L]": preds}

        g = sns.jointplot(
            x="In-situ chl-a concentration [mg/L]",
            y="Forecast chl-a concentration [mg/L]",
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

        
        #g.ax_joint.plot(sns.regplot(x=chl, y=preds, scatter=False, ax=g.ax_joint).lines[0].get_xdata(), sns.regplot(x=chl, y=preds, scatter=False, ax=g.ax_joint).lines[0].get_ydata(), color="black")
        g.ax_joint.plot([min_val, max_val], [min_val, max_val], linestyle="--", color=sns.color_palette("Set2")[2])
        if d != 'a':
            plt.suptitle(f"Day {d}")
        plt.savefig(
            f"{prefix}_histogram.png", bbox_inches="tight", pad_inches=0.01
        )
        plt.show()
        sns.reset_defaults()
        
def generate_residuals_figure(ys_real, num_preds_real, dl_preds_real, prefix, path, i):
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
    print(np.nanmax(num_preds_real))
    print(np.nanmax(dl_preds_real))
    

    sns.kdeplot(data=ys_real, ax=axes[0], label="True", color=my_cmap(1), fill=True)
    sns.kdeplot(data=num_preds_real, ax=axes[0], label="Numerical", color='blue', fill=True)
    sns.kdeplot(data=dl_preds_real, ax=axes[0], label="Deep Learning", color='green', fill=True)
    axes[0].set_xlabel("Chl-a concentration \n [mg/m$^3$]", fontsize=13)
    axes[0].legend(fontsize=11)

    axes[1].scatter(ys_real, np.subtract(num_preds_real, ys_real), alpha=0.2, 
                label="Numerical", color = 'blue')
    axes[1].scatter(ys_real, np.subtract(dl_preds_real, ys_real), alpha=0.2, 
                label="Deep Learning", color = 'green')
    axes[1].set_xlabel("Observed: In-situ", fontsize=13)
    axes[1].legend(fontsize=11)

        
    if i != 'a':
        fig.suptitle(f"Day {i}", fontsize=15)
    # elif i == 'a':
    #     axes[0].set_xlim(-5, 116)
    #     axes[1].set_xlim(-5, 116)
    #     axes[1].set_ylim(-120, 160)
    plt.tight_layout()
    
    
    plt.savefig(
            f"{prefix}_residuals.png", bbox_inches="tight", pad_inches=0.01
        )
    plt.show()
    
#%%
print(rmses_dl_num_dict)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.nanmean(a), scipy.stats.sem(a, nan_policy='omit')
    h = se * scipy.stats.t.ppf((1+confidence)/2., n-1)
    return m, h
#%%
from scipy import stats  
rmses_in_num = []
rmses_in_dl = []
rmses_dl_num = []
r2s_in_num = []
r2s_in_dl = []
r2s_dl_num = []    
for i in range(5):  
    in_temp = np.concatenate(insitu_dict[i])
    num_temp = np.concatenate(num_dict[i])
    dl_temp = np.concatenate(dl_dict[i])
    rmses_dl_num_ = rmses_dl_num_dict[i]
    rmses_in_num_ = rmses_num_in_dict[i]
    rmses_in_dl_ = rmses_dl_in_dict[i]
    
    r2s_dl_num_ = r2s_dl_num_dict[i]
    r2s_in_num_ = r2s_num_in_dict[i]
    r2s_in_dl_ = r2s_dl_in_dict[i]
    
    # generate_pred_histogram(in_temp, num_temp, f"num_insitu_day{i}_GTL", "./Figures", i+1)  
    # generate_pred_histogram(in_temp, dl_temp, f"dl_insitu_day{i}_GTL", "./Figures", i+1) 
    # generate_residuals_figure(in_temp, num_temp, dl_temp, f"chl_{i}_day_total_GTL", "./Figures", i+1)   
    
    print("Day", i+1)
    
   
    print(mean_confidence_interval(rmses_in_num_))
    print(mean_confidence_interval(rmses_in_dl_))
    print(mean_confidence_interval(rmses_dl_num_))
    
    
    print(np.nanstd(rmses_in_num_))
    print(np.nanstd(rmses_in_dl_))
    print(np.nanstd(rmses_dl_num_))
    
    
    print(np.nanmean(r2s_in_num_))
    print(np.nanmean(r2s_in_dl_))
    print(np.nanmean(r2s_dl_num_))
    

    # print(np.corrcoef(in_temp, num_temp))
    # print(np.corrcoef(in_temp, dl_temp))
    # print(np.corrcoef(num_temp, dl_temp))
    
    # #print("NUMERICAL - INSITU")
    # abs_diff = np.abs(num_temp - in_temp)
    # rmse = np.sqrt(np.nanmean(abs_diff**2))
    
    # r2 = R_squared(in_temp, num_temp)
    # print("RMSE: ")
    # print(rmse)
    # print("R2: ")
    # print(r2)

    # rmses_in_num.append(rmse)
    # r2s_in_num.append(r2)
    
    # print("DL - INSITU")
    # abs_diff = np.abs(dl_temp - in_temp)
    # rmse = np.sqrt(np.nanmean(abs_diff**2))
    
    # r2 = R_squared(in_temp, dl_temp)
    # print("RMSE: ")
    # print(rmse)
    # print("R2: ")
    # print(r2)

    # rmses_in_dl.append(rmse)
    # r2s_in_dl.append(r2)
    
    # print("NUMERICAL - DL")
    # abs_diff = np.abs(chl_dl_GTL_total[:,t,:,:] - chl_num_GTL_total[:,t,:,:])
    # rmse = np.sqrt(np.nanmean(abs_diff**2))
    
    # r2 = R_squared(chl_dl_GTL_total[:,t,:,:].flatten(), chl_num_GTL_total[:,t,:,:].flatten())
    # print("RMSE: ")
    # print(rmse)
    # print("R2: ")
    # print(r2)

    # rmses_dl_num.append(rmse)
    # r2s_dl_num.append(r2)
    
insitu_filtered = np.concatenate([insitu_dict[i] for i in range(5)], axis=0)
chl_num_GTL_filtered = np.concatenate([num_dict[i] for i in range(5)], axis=0)
chl_dl_GTL_filtered = np.concatenate([dl_dict[i] for i in range(5)], axis=0)

insitu_filtered = np.concatenate(insitu_filtered, axis=0)
chl_num_GTL_filtered = np.concatenate(chl_num_GTL_filtered, axis=0)
chl_dl_GTL_filtered = np.concatenate(chl_dl_GTL_filtered, axis=0)

print([r2s_dl_in_dict[i] for i in range(5)])
r2s_dl_in = np.concatenate([r2s_dl_in_dict[i] for i in range(5)], axis=0)
r2s_num_in = np.concatenate([r2s_num_in_dict[i] for i in range(5)], axis=0)
r2s_dl_num = np.concatenate([r2s_dl_num_dict[i] for i in range(5)], axis=0)

rmses_dl_in = np.concatenate([rmses_dl_in_dict[i] for i in range(5)], axis=0)
rmses_num_in = np.concatenate([rmses_num_in_dict[i] for i in range(5)], axis=0)
rmses_dl_num = np.concatenate([rmses_dl_num_dict[i] for i in range(5)], axis=0)


print("NUMERICAL - INSITU")  
abs_diff = np.abs(chl_num_GTL_filtered - insitu_filtered)
print("RMSE: ")
print(mean_confidence_interval(rmses_num_in))
print("R2: ")
print(mean_confidence_interval(r2s_num_in))

print("DL - INSITU")  
abs_diff = np.abs(chl_dl_GTL_filtered - insitu_filtered)
print("RMSE: ")
print(mean_confidence_interval(rmses_dl_in))
print("R2: ")
print(mean_confidence_interval(r2s_dl_in))

print("DL - NUMERICAL")  
abs_diff = np.abs(chl_dl_GTL_total - chl_num_GTL_total)
print("RMSE: ")
print(mean_confidence_interval(rmses_dl_num))
print("R2: ")
print(mean_confidence_interval(r2s_dl_num))
    
#%%
    


print(insitu_filtered[0])
print(np.min(insitu_filtered))
generate_pred_histogram(insitu_filtered, chl_num_GTL_filtered, f"num_insitu_all", "./Figures", 'a')
generate_residuals_figure(insitu_filtered, chl_num_GTL_filtered, chl_dl_GTL_filtered, f"num_insitu_all", "./Figures", 'a')

#%%
generate_pred_histogram(insitu_filtered, chl_dl_GTL_filtered, f"insitu_dl_all", "./Figures", 'a')

#%%
generate_pred_histogram(chl_num_GTL_total.flatten(), chl_dl_GTL_total.flatten(), f"num_dl_all_GTL", "./Figures", 'a') 


#%%
sat_dir = r"C:\Users\nioh\Documents\MasterThesis\data\InSitu\Features"

sat_obs_GTL = np.load(f"{sat_dir}/True_CHL_GTL.npy")
#%%
n = 3
fig, axes = plt.subplots(3, 5, figsize=(12, 6))

for i in range(5):
    im1 = axes[0, i].imshow(sat_obs_GTL[n,i,:,:], origin='lower', vmin=0, vmax=5)
    im2 = axes[1, i].imshow(chl_num_GTL_total[n,i,:,:], origin='lower', vmin=0, vmax=5)
    im3 = axes[2, i].imshow(chl_dl_GTL_total[n,i,:,:], origin='lower', vmin=0, vmax=5)
    
    axes[0, i].tick_params(labelbottom=False, labelleft=False, length=0)
    axes[1, i].tick_params(labelbottom=False, labelleft=False, length=0)
    axes[2, i].tick_params(labelbottom=False, labelleft=False, length=0)
    axes[0, i].set_title(f"Predicted day {i+1}")
axes[0, 0].set_ylabel("Satellite \nImage", fontsize=13)
axes[1, 0].set_ylabel("Numerical \nForecast", fontsize=13)
axes[2, 0].set_ylabel("Deep Learning \nPrediction", fontsize=13)
plt.colorbar(im1, ax=axes[0, :].tolist(), pad=0.01)
plt.colorbar(im2, ax=axes[1, :].tolist(), pad=0.01)
plt.colorbar(im3, ax=axes[2, :].tolist(), pad=0.01)
plt.show()

        
#%%        
plt.figure()
plt.scatter(range(len(chl_num_GTL_filtered)),chl_num_GTL_filtered)
plt.scatter(range(len(insitu_filtered)),insitu_filtered)
plt.title("Predictions vs Ferrybox Insitu measurements")
plt.legend(["Numerical Forecast", "Insitu"])

# %%
########KATTEGAT / RIGA#########
# open the insitu measurements saved in a grid
insitu_file_1 = r"C:\Users\nioh\Documents\MasterThesis\data\InSitu\gtl_ferrybox_measurements_2062023_to_2562023.npy"
insitu_file_2 = r"C:\Users\nioh\Documents\MasterThesis\data\InSitu\gtl_ferrybox_measurements_2662023_to_2862023.npy"
insitu_1 = np.load(insitu_file_1)
insitu_2 = np.load(insitu_file_2)

print(insitu_1.shape)
print(insitu_2.shape)

insitu_total = np.concatenate((insitu_1, insitu_2), axis=0)
print(insitu_total.shape)

#%%
# open the numerical data output

num_dir = r"C:\Users\nioh\Documents\MasterThesis\data\InSitu\CHL_Numerical"
lat_lon_dir = r"C:\Users\nioh\Documents\MasterThesis\data\Numerical_Model_Data\CHL_data_21122021_to_1012022.nc"
df_lat_lon = xr.open_dataset(lat_lon_dir)
chl_data_init = df_lat_lon.sel(time=~df_lat_lon.get_index("time").duplicated())
chl_data_init = chl_data_init.rename_dims({'lon': 'x', 'lat': 'y'}).rename_vars({"lon": "x", "lat": "y"}).set_coords(["x", "y"])
lat_lon_subset = chl_data_init[['x', 'y']]

chl_num__total = np.load(f"{num_dir}/num_forecast_KTT.npy")

#%%
# open the deep learning model output
deep_dir = r"C:\Users\nioh\Documents\MasterThesis\data\InSitu\CHL_Satellite"

chl_dl__total = np.load(f"{deep_dir}/deep_learning_preds_KTT.npy")
print(chl_dl__total.shape)


# %%

num_dict = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: []
}

dl_dict = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: []
}



rmses_dl_num_dict = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: []
}


r2s_dl_num_dict = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: []
}


def R_squared(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0,1]
    R_sq = corr**2
    return R_sq

for i in range(5):
    print("EXAMPLE DAY: ", i)
    # adjust the dates to the numerical model output predictions
    chl_num_ = chl_num__total[i,:,:,:]
    chl_dl_ = chl_dl__total[i,:,:,:]


    # compare the insitu measurements with the numerical model output
    # for each day
    rmses_dl_num = []
    r2s_dl_num = []

    for t in range(5):
        print(f"Day: {t}")
        num = chl_num_[t,:,:]
        dl = chl_dl_[t,:,:]
        
        print("NUMERICAL - DL")
        abs_diff = np.abs(chl_dl_[t,:,:] - chl_num_[t,:,:])
        rmse = np.sqrt(np.nanmean(abs_diff**2))
        
        r2 = R_squared(dl, num)
        print("RMSE: ")
        print(rmse)
        print("R2: ")
        print(r2)
        rmses_dl_num_dict[t].append(rmse)
        r2s_dl_num_dict[t].append(r2)
        rmses_dl_num.append(rmse)
        r2s_dl_num.append(r2)
    
    print("DL - NUMERICAL")  
    abs_diff = np.abs(chl_dl_ - chl_num_)
    print("RMSE: ")
    print(mean_confidence_interval(rmses_dl_num))
    print("R2: ")
    print(mean_confidence_interval(r2s_dl_num))
# %%

#%%
def generate_pred_histogram(chl, preds, prefix, path, d, fig_height=4):
        sns.set_theme(style="whitegrid")
        print("Start generating histogram...")
        min_val = np.min([np.min(chl), np.min(preds)])-0.3
        max_val = np.max([np.max(chl), np.max(preds)])+0.3
        
        data = {"Numerical chl-a concentration [mg/L]": chl, "Deep learning chl-a concentration [mg/L]": preds}

        g = sns.jointplot(
            x="Numerical chl-a concentration [mg/L]",
            y="Deep learning chl-a concentration [mg/L]",
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

        
        #g.ax_joint.plot(sns.regplot(x=chl, y=preds, scatter=False, ax=g.ax_joint).lines[0].get_xdata(), sns.regplot(x=chl, y=preds, scatter=False, ax=g.ax_joint).lines[0].get_ydata(), color="black")
        g.ax_joint.plot([min_val, max_val], [min_val, max_val], linestyle="--", color=sns.color_palette("Set2")[2])
        if d != 'a':
            plt.suptitle(f"Day {d}")
        plt.savefig(
            f"{prefix}_histogram.png", bbox_inches="tight", pad_inches=0.01
        )
        plt.show()
        sns.reset_defaults()
        
def generate_residuals_figure(ys_real, num_preds_real, dl_preds_real, prefix, path, i):
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
    print(np.nanmax(num_preds_real))
    print(np.nanmax(dl_preds_real))
    

    sns.kdeplot(data=ys_real, ax=axes[0], label="True", color=my_cmap(1), fill=True)
    sns.kdeplot(data=num_preds_real, ax=axes[0], label="Numerical", color='blue', fill=True)
    sns.kdeplot(data=dl_preds_real, ax=axes[0], label="Deep Learning", color='green', fill=True)
    axes[0].set_xlabel("Chl-a concentration \n [mg/m$^3$]", fontsize=13)
    axes[0].legend(fontsize=11)

    axes[1].scatter(ys_real, np.subtract(num_preds_real, ys_real), alpha=0.2, 
                label="Numerical", color = 'blue')
    axes[1].scatter(ys_real, np.subtract(dl_preds_real, ys_real), alpha=0.2, 
                label="Deep Learning", color = 'green')
    axes[1].set_xlabel("Observed: In-situ", fontsize=13)
    axes[1].legend(fontsize=11)

        
    if i != 'a':
        fig.suptitle(f"Day {i}", fontsize=15)
    # elif i == 'a':
    #     axes[0].set_xlim(-5, 116)
    #     axes[1].set_xlim(-5, 116)
    #     axes[1].set_ylim(-120, 160)
    plt.tight_layout()
    
    
    plt.savefig(
            f"{prefix}_residuals.png", bbox_inches="tight", pad_inches=0.01
        )
    plt.show()
    
def generate_dist_figure(num_preds_real, dl_preds_real, prefix, path, i):
    # Define colormap
    cm = plt.cm.get_cmap('viridis')
    my_cmap = cm(np.linspace(0,1,10))

    # Create new colormap
    my_cmap = ListedColormap(my_cmap)
    fig, axes = plt.subplots(1, 1, figsize=(4, 6))
    axes.set_ylabel("Density", fontsize=13) 

    axes.grid()

    sns.kdeplot(data=num_preds_real[np.where(dl_preds_real<25)], ax=axes, label="Numerical", color='blue', fill=True)
    sns.kdeplot(data=dl_preds_real[np.where(dl_preds_real<25)], ax=axes, label="Deep Learning", color='green', fill=True)
    axes.set_xlabel("Chl-a concentration \n [mg/m$^3$]", fontsize=13)
    axes.legend(fontsize=11)
        
    if i != 'a':
        fig.suptitle(f"Day {i}", fontsize=15)
    # elif i == 'a':
    #     axes[0].set_xlim(-5, 116)
    #     axes[1].set_xlim(-5, 116)
    #     axes[1].set_ylim(-120, 160)
    plt.tight_layout()
    
    
    plt.savefig(
            f"{prefix}_distribution.png", bbox_inches="tight", pad_inches=0.01
        )
    plt.show()
    
#%%
print(rmses_dl_num_dict)

#%%

rmses_dl_num = []
r2s_dl_num = []    
for i in range(5):  

    rmses_dl_num_ = rmses_dl_num_dict[i]
    
    r2s_dl_num_ = r2s_dl_num_dict[i]
    r2s_in_num_ = r2s_num_in_dict[i]
    r2s_in_dl_ = r2s_dl_in_dict[i]
    
    # generate_pred_histogram(in_temp, num_temp, f"num_insitu_day{i}_", "./Figures", i+1)  
    # generate_pred_histogram(in_temp, dl_temp, f"dl_insitu_day{i}_", "./Figures", i+1) 
    generate_dist_figure(chl_num__total[:,t,:,:].flatten(), chl_dl__total[:,t,:,:].flatten(), f"chl_{i}_day_total_RIGA", "./Figures", i+1)   
    
    print("Day", i+1)

    print(mean_confidence_interval(rmses_dl_num_))

    print(np.nanstd(rmses_dl_num_))

    print(np.nanmean(r2s_dl_num_))


print([r2s_dl_in_dict[i] for i in range(5)])
r2s_dl_num = np.concatenate([r2s_dl_num_dict[i] for i in range(5)], axis=0)
rmses_dl_num = np.concatenate([rmses_dl_num_dict[i] for i in range(5)], axis=0)


print("DL - NUMERICAL")  
abs_diff = np.abs(chl_dl__total - chl_num__total)
print("RMSE: ")
print(mean_confidence_interval(rmses_dl_num))
print("R2: ")
print(np.nanmean(r2s_dl_num))
    
#%%
generate_dist_figure(chl_num__total.flatten(), chl_dl__total.flatten(), f"chl_total_", "./Figures", 'a')  
generate_pred_histogram(chl_num__total[~np.isnan(chl_num__total)].flatten(), chl_dl__total[~np.isnan(chl_num__total)].flatten(), f"dl_num_RIGA", "./Figures", 'a') 



#%%
sat_dir = r"C:\Users\nioh\Documents\MasterThesis\data\InSitu\Features"
landmask = np.load(f"{sat_dir}/landmask_RIGA.npy")
sat_obs = np.load(f"{sat_dir}/True_CHL_RIGA.npy")
#%%
n = 2
fig, axes = plt.subplots(3, 5, figsize=(12, 6))

for i in range(5):
    im1 = axes[0, i].imshow(sat_obs[n,i,:,:], origin='lower', vmin=0, vmax=40)
    im2 = axes[1, i].imshow(chl_num__total[n,i,:,:], origin='lower', vmin=0, vmax=4)
    im3 = axes[2, i].imshow(np.where(landmask==1,np.nan,chl_dl__total[n,i,:,:]), origin='lower', vmin=0, vmax=40)
    
    axes[0, i].tick_params(labelbottom=False, labelleft=False, length=0)
    axes[1, i].tick_params(labelbottom=False, labelleft=False, length=0)
    axes[2, i].tick_params(labelbottom=False, labelleft=False, length=0)
    axes[0, i].set_title(f"Predicted day {i+1}")
axes[0, 0].set_ylabel("Satellite \nImage", fontsize=13)
axes[1, 0].set_ylabel("Numerical \nForecast", fontsize=13)
axes[2, 0].set_ylabel("Deep Learning \nPrediction", fontsize=13)
plt.colorbar(im1, ax=axes[0, :].tolist(), pad=0.01)
plt.colorbar(im2, ax=axes[1, :].tolist(), pad=0.01)
plt.colorbar(im3, ax=axes[2, :].tolist(), pad=0.01)
plt.show()

        
#%%        
plt.figure()
plt.scatter(range(len(chl_num_GTL_filtered)),chl_num_GTL_filtered)
plt.scatter(range(len(insitu_filtered)),insitu_filtered)
plt.title("Predictions vs Ferrybox Insitu measurements")
plt.legend(["Numerical Forecast", "Insitu"])