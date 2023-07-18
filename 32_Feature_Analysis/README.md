# Feature Correlation Analysis

The notebook contains the code to perform the feature correlation analysis and cluster the features based on hierarchical clustering.

It must be ensured that the following files are present:

- a .nc file containing a longer time-series of initial (not gap-filled) CHL values with a variable "CHL"
- a .nc file containing a longer time-series of gap-filled CHL variables, with one variable containing the gap-filled CHL data and one variable containing a landmask
- a file "wind_v10.npy" containing the data of the Wind_v10 for the exact time period
- a file "wind_u10.npy" containing the data of the Wind_u10 for the exact time period
- a file "pres_t2m.npy containing data of the prescipitation for the exact time period
- a file "pres_tp.npy" containing data of the air temperature for the exact time period
- a file "pres_rad.npy" containing data of the solar radiation for the exact time period
- a file "sst_values.npy" containing data of the sea surface temperature for the exact time period
- a file "VHM0_data.npy" containing data of wave height for the exact time period
- a file "VMDR_data.npy" containing data of wave direction for the exact time period
- a file "topo.npy" containing data of topography of the area