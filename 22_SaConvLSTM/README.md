# Sa-ConvLSTM

The Sa-ConvLSTM layer that is used in this model follows the method explained by Lin et al. https://ojs.aaai.org//index.php/AAAI/article/view/6819 and uses the proposed layer: https://github.com/MahatmaSun1/SaConvSLTM/blob/041ecb020d151a21b4a1c3426c2e4e56269c5bff/SaConvSLTM/SaConvLSTM.py . 

If it is ensured that the required libraries are installed and training data is present in a seperate directory, the model can be trained. The training data must contain:

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




The code can be used to train the model and generate new predictions by:

1. Adjust *interpolated_value*, *DATA_PATH_CHL*, *DATA_PATH_CHL_INIT*, *DATA_PATH_CHL_Features* in the 01_train.py file to name the filed variable and define the training data directories
2. Adjust the model name in the 01_train.py, 02_predict.py file according to the requirements
3. Run the 01_train.py file (in terminal: python3 -m 01_train.py)
4. Adjust the optimal number of epochs in the 02_predict.py file
5. Run the 02_predict.py file (in terminal: python3 -m 02_predict.py)

Step 3. saves the weights of the model in every iteration that the model performed better than in any previous run. These weights are then called in step 5. to perform the predictions and immediately run the validation, calculating the RMSE and $R^2$ values for each day and generating plots. 