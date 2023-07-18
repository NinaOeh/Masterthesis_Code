# Comparisons

This folder holds the code for the comparison of the model output to interpolation as well as in-situ measurements. 

1. Comparison to interpolation techniques: For the comparison to interpolation techniques, it must be ensured that the gap-filled data is saved in individual files and the corresponding variable *init_path* in the file "Interpolation_methods.py" points to the repository that contains those files.

2. Comparison to numerical and in-situ data: In order to use the provided code, it must be ensured that the in-situ measurements and numerical as well as deep-learning predictions are saved in seperate files and the exact dates for each file must be known. The code must be adjusted according to these days and point to the correct repositories for all data.