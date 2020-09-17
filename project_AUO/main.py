# -*- coding: utf-8 -*-

from util.util import *
from util.ms_util import *
from util.ceemdae_util import *

#for window
#DATA_PATH = "C:/Users/user/Desktop/all_proj/AUO/107_kw.csv"

#for Mac
DATA_PATH ="./data/107_kw.csv"

#================for global===============#
INPUT_LEN = 96
OUTPUT_LEN = 24
DAILY_TIME_STEP = 96
#==================for MS=================#
FIL_SEQ_NUM = 10
DOWN_SAM_NUM = 4 #must be divisor of INPUT_LEN

#===============for CEEMD-AE==============#


if __name__ == '__main__': 
    
    #MS
    fil_data = load_data(DATA_PATH,0,True)
    Pro_data , Tar_dataset = resample_data(fil_data,INPUT_LEN,DAILY_TIME_STEP)
    MS_dataset = get_ms_dataset(Pro_data , FIL_SEQ_NUM , DOWN_SAM_NUM)

    #CEEMDAE
    #fil_data = load_data(DATA_PATH,0,True)
    #Pro_data , Tar_dataset = resample_data(fil_data,INPUT_LEN,DAILY_TIME_STEP)
    #IMFs = get_IMFset(Pro_data)
    #grop_data = excute_ae(IMFs)









