#%%
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from tqdm import tqdm

#%%
# ferry box: 
#file = r"C:\Users\nioh\Documents\MasterThesis\data\copernicus_BAL_insitu_nrt_FB_d488_8a07_0fbf.nc"
# stationary stations:
#file = r"C:\Users\nioh\Documents\MasterThesis\data\copernicus_BAL_insitu_nrt_MO_171d_b6a7_9542.nc"
# newest fery box:
file = r"C:\Users\nioh\Documents\MasterThesis\data\InSitu\ferrybox_insitu_2062023_to_2562023.nc"
mask_dir = r"C:\Users\nioh\Documents\MasterThesis\data\mask.nc"


#%%
df = xr.open_dataset(file)
df = df.to_dataframe()
print(df.columns)
# %%
df_chl = df[df['FLU2'] != None]
df_chl = df_chl[df_chl['FLU2'] != 'NaN']
df_chl = df_chl[df_chl['FLU2'] > 0]

# df_chl = df[df['CHLT'] != None]
# df_chl = df_chl[df_chl['CHLT'] != 'NaN']
# %%
print(df_chl['PLATFORM_NAME'].unique())
# %%
lats = df_chl['latitude'].unique()
lons = df_chl['longitude'].unique()

print(len(lats), len(lons))
# %%
station_locations = df_chl.groupby(['latitude','longitude']).size().reset_index().rename(columns={0:'count'})
station_locations
lons = station_locations['longitude'][:-1]
lats = station_locations['latitude'][:-1]
s=station_locations['count'][:-1]
# %%
# open mask
map_data= xr.open_dataset(f"{mask_dir}")

#%%
map_data
llcrnrlon = map_data.XT_I.min()
llcrnrlat = map_data.YT_J.min()
urcrnrlon = map_data.XT_I.max()
urcrnrlat = map_data.YT_J.max()
# %%
mask = map_data['LANDMASK'].values
# %%
# Create a figure and axes for the plot
fig, ax = plt.subplots(figsize=(6,6))

# Plot the map data using pcolormesh
map_data.LANDMASK.plot.pcolormesh(ax=ax, cmap='Greys',
                         x='XT_I', y='YT_J', add_colorbar=False)

# Plot the points on the map using scatter
ax.scatter(lons, lats, s=20, c='yellow', edgecolors='green', label='Ferrybox measurements')

# lat_rg = np.linspace(56.88, 58.63, 160)
# lon_rg = np.linspace(22.02, 24.84, 160)
# lat_kg = np.linspace(55.45, 57.2, 160)
# lon_kg = np.linspace(10.14, 12.96, 160)
# lat_gt = np.linspace(55.01, 56.76, 160)
# lon_gt = np.linspace(17.59, 20.41, 160)

coord_dk = [[10.14,55.45], [10.14,57.2], [12.96,57.2], [12.96,55.45]]
coord_dk.append(coord_dk[0])
xs_dk, ys_dk = zip(*coord_dk)
# area of golf of riga
coord_rg = [[22.02, 56.88], [24.84, 56.88], [24.84, 58.63], [22.02, 58.63]]
coord_rg.append(coord_rg[0])
xs_rg, ys_rg = zip(*coord_rg)
# area of bothnian sea
coord_bb = [[17.59,55.01], [17.59,56.76], [20.41,56.76], [20.41,55.01]]
coord_bb.append(coord_bb[0])
xs_bb, ys_bb = zip(*coord_bb)

ax.plot(xs_dk,ys_dk, color='red', label='Southern Kattegat & Danish Belts')
ax.plot(xs_rg,ys_rg, color='orange', label='Gulf of Riga')
ax.plot(xs_bb,ys_bb, color='blue', label='South-Eastern Gotland Basin')

# Set the axis limits to the geographic bounds of the map image
ax.set_xlim(llcrnrlon, urcrnrlon)
ax.set_ylim(llcrnrlat, urcrnrlat)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Add a title and show the plot
ax.set_title('Available Ferrybox measurements of \n chlorophyll fluorescence data from 20.06.2023 to 28.06.2023')
ax.legend()
plt.show()

# %%

#FLU2
print(np.nanmin(df_chl['FLU2'][2:]), np.nanmax(df_chl['FLU2'][2:]))
#CHLT
#print(np.nanmin(df_chl['CHLT'][2:]), np.nanmax(df_chl['CHLT'][2:]))

# %%
# take daily mean of chl-a per station
df_chl['time'] = pd.to_datetime(df_chl['time'])
df_chl['date'] = df_chl['time'].dt.date
df_chl['date'] = pd.to_datetime(df_chl['date'])
df_chl

#%%
df_chl['DEPH'].unique()
#%%
# check for numeric values
[(c, df_chl[c].dtype.kind in 'iufcb') for c in df_chl.columns]

#%%
# drop columns with non-numeric values
df_chl_daily = df_chl[['date','latitude','longitude','FLU2', 'FLU2_QC']]
# %%
df_chl_daily = df_chl_daily.groupby(['date','latitude','longitude']).mean().reset_index()
df_chl_daily

# %%
# df_chl gulf of riga
df_chl_daily_gr = df_chl_daily[(df_chl_daily['latitude'] > 56.5) & (df_chl_daily['latitude'] < 59) & (df_chl_daily['longitude'] > 22) & (df_chl_daily['longitude'] < 25)]
df_chl_daily_gr
# %%
# df_chl kattegat
df_chl_daily_kg = df_chl_daily[(df_chl_daily['latitude'] > 55.4) & (df_chl_daily['latitude'] < 57.4) & (df_chl_daily['longitude'] > 10) & (df_chl_daily['longitude'] < 13)]
df_chl_daily_kg = df_chl_daily_kg[df_chl_daily_kg['FLU2'] > 0]
df_chl_daily_kg

# %%
# df_chl gotland
df_chl_daily_gt = df_chl_daily[(df_chl_daily['latitude'] > 55) & (df_chl_daily['latitude'] < 57) & (df_chl_daily['longitude'] > 17.5) & (df_chl_daily['longitude'] < 20.5)]
df_chl_daily_gt = df_chl_daily_gt[df_chl_daily_gt['FLU2'] > 0]
df_chl_daily_gt

#%%
# count the rows per day
df_chl_daily_gt['date'].value_counts()

# %%
# create a grid with 160 values between 53.26 and 65.84 in Y direction (lat) and 160 values between 9.259 and 30.24 in X direction (lon)
# this is the grid of the model

# Riga 
lat_rg = np.linspace(56.88, 58.63, 160)
lon_rg = np.linspace(22.02, 24.84, 160)
lat_kg = np.linspace(55.45, 57.2, 160)
lon_kg = np.linspace(10.14, 12.96, 160)
lat_gt = np.linspace(55.01, 56.76, 160)
lon_gt = np.linspace(17.59, 20.41, 160)

#time_range = pd.date_range(start='2023-05-06', end='2023-05-31', freq='D')
time_range = pd.date_range(start='2023-06-26', end='2023-06-28', freq='D')
time_range = time_range.map(lambda t: t.strftime('%Y-%m-%d'))
print(time_range)
print(lat_kg)
print(lon_kg)

df_chl_daily['date'] = df_chl_daily['date'].map(lambda t: t.strftime('%Y-%m-%d'))
#%%
t=3
print(df_chl_daily[(df_chl_daily['date'] == time_range[t]) & (df_chl_daily['latitude'] >= lat_kg[0]) & (df_chl_daily['latitude'] <= lat_kg[-1]) & (df_chl_daily['longitude'] >= lon_kg[0]) & (df_chl_daily['longitude'] <= lon_kg[-1]) & (df_chl_daily['FLU2'] > 0)]['FLU2'].mean())
#%%
# calculate the daily average for the data within one grid cell
# Kattegat
ktt_measurements = np.empty((len(time_range),160, 160))
print(len(time_range))
for t in tqdm(range(len(time_range))):
    for i in range(len(lat_rg)):
        for j in range(len(lon_rg)): 
            #print(t,i,j)
            if i == len(lat_rg)-1 and j != len(lon_rg)-1:
                flu = df_chl_daily[(df_chl_daily['latitude'] >= lat_kg[i]) & (df_chl_daily['latitude'] <= 57.2) & (df_chl_daily['longitude'] >= lon_kg[j]) & (df_chl_daily['longitude'] <= lon_kg[j+1]) & (df_chl_daily['date'] == time_range[t]) & (df_chl_daily['FLU2'] > 0)]['FLU2'].mean()
            elif j == len(lon_rg)-1 and i != len(lat_rg)-1 :
                flu = df_chl_daily[(df_chl_daily['latitude'] >= lat_kg[i]) & (df_chl_daily['latitude'] <= lat_kg[i+1]) & (df_chl_daily['longitude'] >= lon_kg[j]) & (df_chl_daily['longitude'] <= 12.96) & (df_chl_daily['date'] == time_range[t]) & (df_chl_daily['FLU2'] > 0)]['FLU2'].mean()
            elif j == len(lon_rg)-1 and i == len(lat_rg)-1 :
                flu = df_chl_daily[(df_chl_daily['latitude'] >= lat_kg[i]) & (df_chl_daily['latitude'] <= 57.2) & (df_chl_daily['longitude'] >= lon_kg[j]) & (df_chl_daily['longitude'] <= 12.96) & (df_chl_daily['date'] == time_range[t]) & (df_chl_daily['FLU2'] > 0)]['FLU2'].mean()
            else:
                flu = df_chl_daily[(df_chl_daily['latitude'] >= lat_kg[i]) & (df_chl_daily['latitude'] <= lat_kg[i+1]) & (df_chl_daily['longitude'] >= lon_kg[j]) & (df_chl_daily['longitude'] <= lon_kg[j+1]) & (df_chl_daily['date'] == time_range[t]) & (df_chl_daily['FLU2'] > 0)]['FLU2'].mean()
            
            if flu > 0:
                ktt_measurements[t,j,i] = flu
                print(flu)
            else:
                ktt_measurements[t,j,i] = np.nan
#%%              
# Gotland
gtl_measurements = np.empty((len(time_range),160, 160))
print(len(time_range))
for t in tqdm(range(len(time_range))):
    for i in range(len(lat_rg)):
        for j in range(len(lon_rg)): 
            #print(t,i,j)
            if i == len(lat_rg)-1 and j != len(lon_rg)-1:
                flu = df_chl_daily[(df_chl_daily['latitude'] >= lat_gt[i]) & (df_chl_daily['latitude'] <= 56.76) & (df_chl_daily['longitude'] >= lon_gt[j]) & (df_chl_daily['longitude'] <= lon_gt[j+1]) & (df_chl_daily['date'] == time_range[t]) & (df_chl_daily['FLU2'] > 0)]['FLU2'].mean()
            elif j == len(lon_rg)-1 and i != len(lat_rg)-1 :
                flu = df_chl_daily[(df_chl_daily['latitude'] >= lat_gt[i]) & (df_chl_daily['latitude'] <= lat_gt[i+1]) & (df_chl_daily['longitude'] >= lon_gt[j]) & (df_chl_daily['longitude'] <= 20.41) & (df_chl_daily['date'] == time_range[t]) & (df_chl_daily['FLU2'] > 0)]['FLU2'].mean()
            elif j == len(lon_rg)-1 and i == len(lat_rg)-1 :
                flu = df_chl_daily[(df_chl_daily['latitude'] >= lat_gt[i]) & (df_chl_daily['latitude'] <= 56.76) & (df_chl_daily['longitude'] >= lon_gt[j]) & (df_chl_daily['longitude'] <= 20.41) & (df_chl_daily['date'] == time_range[t]) & (df_chl_daily['FLU2'] > 0)]['FLU2'].mean()
            else:
                flu = df_chl_daily[(df_chl_daily['latitude'] >= lat_gt[i]) & (df_chl_daily['latitude'] <= lat_gt[i+1]) & (df_chl_daily['longitude'] >= lon_gt[j]) & (df_chl_daily['longitude'] <= lon_gt[j+1]) & (df_chl_daily['date'] == time_range[t]) & (df_chl_daily['FLU2'] > 0)]['FLU2'].mean()
            
            if flu > 0:
                gtl_measurements[t,j,i] = flu
                print(flu)
            else:
                gtl_measurements[t,j,i] = np.nan
            
#%%
np.count_nonzero(~np.isnan(gtl_measurements))

#%%
for t in range(len(time_range)):
    plt.imshow(gtl_measurements[1,:,:])
    plt.show()
    
    
#%%
print(gtl_measurements[~np.isnan(gtl_measurements)])
    
#%%
np.save(r"C:\Users\nioh\Documents\MasterThesis\data\InSitu\gtl_ferrybox_measurements_2662023_to_2862023.npy", gtl_measurements)
#%%
# open the numerical data output

num_dir = r"C:\Users\nioh\Documents\MasterThesis\data\Numerical_Model_Data\CHL_NumForecast_data_352023_to_3052023.nc"
df_num = xr.open_dataset(num_dir)
df_num

#%%
df_num = df_num.to_dataframe()
df_num
# %%
