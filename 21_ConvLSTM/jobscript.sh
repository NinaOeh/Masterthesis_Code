#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
### -- set the job Name -- 
#BSUB -J LSTMminmax_init
### -- ask for number of cores (default: 1) -- 
###BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=40GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 3GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Output_%J.err 

# here follow the commands you want to execute with input.in as the input file
module swap python3/3.10.7
nvidia-smi
# Load the cuda module
module load cuda/11.6
module load cudnn/v8.3.2.44-prod-cuda-11.X 
module load tensorrt/7.2.3.4-cuda-11.X 


echo "Hello World" 
python3 01_train.py
#python3 04_tune.py