# STpConv (pConv3D)

The code in this repository is based and partly copied from the pConv3D neural netwrok implementation of Marius Appel, that is used to fill gaps in timeseries of satellite data. Details to the inital model are described in this paper: https://arxiv.org/abs/2208.08781 and the implementation of this github repository: https://github.com/appelmar/STpconv. 

The model was modified from the initial model, adding more layers and different skip connections. If it is ensured that the required libraries are installed and training data is present in a seperate directory, consisting of .nc files with 20-day periods of CHL data, the model can be used to train the model and generate new predictions by:

1. Adjust *DATA_PATH_TRAINING* and *DATA_PATH_VALIDATION* in the 01_train_chl.py file to point to the training data directory
2. Adjust the model name in the 01_train_chl.py, 02_predict_chl.py and 03_validation_chl.py file according to the requirements
3. Run the 01_train_chl.py file (in terminal: python3 -m 01_train_chl.py)
4. Adjust the optimal number of epochs in the 02_predict_chl.py file
5. Run the 02_predict_chl.py file (in terminal: python3 -m 02_predict_chl.py)
6. Run the 03_validation_chl.py file (in terminal: python3 -m 03_validation_chl.py)

Step 3. saves the weights of the model in every iteration that the model performed better than in any previous run. These weights are then called in step 5. to perform the predictions, saving them seperately for every 20-day set of the test dataset in a new .tif file. Step 6. runs the validation, calculating the RMSE and $R^2$ values and generating plots. 

To fill a complete time-series with multiple years of data based on a pre-trained model the 02_predict_all.py file can be used.