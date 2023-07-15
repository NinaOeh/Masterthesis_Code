# Extension of DINCAE v2 model, with partial convolutions in first U-NET


DINCAE (Data-Interpolating Convolutional Auto-Encoder, Alexander Bath et al) is a neural network to
reconstruct missing data in satellite observations. Two versions of DINCAE have been released of which the first (DINCAEv1) is described in the following open access paper: https://doi.org/10.5194/gmd-13-1609-2020 and has been released using tensorflow python on this gihub page: https://github.com/gher-uliege/DINCAE and the second version (DINCAEv2) is described in this paper: https://doi.org/10.5194/gmd-15-2183-2022 and has been released using julia on this gthub page: https://github.com/gher-ulg/DINCAE.jl*.

In this reporsitory the existing DINCAEv2 model has been extended and partial convolutions have been applied in the first U-NET of the model. 

If it is ensured that the required libraries are installed and training data is present in a seperate directory, consisting of .nc files with 20-day periods of CHL data, the model can be used to train the model and generate new predictions by:

1. Adjust *DATA_PATH_TRAINING* and *DATA_PATH_VALIDATION* in the 01_ptrain.py file to point to the training data directory
2. Adjust the model name in the 01_ptrain.py, 02_ppredict.py and 03_pvalidation.py file according to the requirements
3. Run the 01_ptrain.py file (in terminal: python3 -m 01_ptrain.py)
4. Adjust the optimal number of epochs in the 02_ppredict.py file
5. Run the 02_ppredict.py file (in terminal: python3 -m 02_ppredict.py)
6. Run the 03_pvalidation.py file (in terminal: python3 -m 03_pvalidation.py)

Step 3. saves the weights of the model in every iteration that the model performed better than in any previous run. These weights are then called in step 5. to perform the predictions, saving them seperately for every 20-day set of the test dataset in a new .tif file. Step 6. runs the validation, calculating the RMSE and $R^2$ values and generating plots. 