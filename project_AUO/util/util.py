# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn import svm , preprocessing


def load_data (filename , indexname , filt = False):
    
    print("start loading data...")

    raw_data = pd.read_csv(filename , index_col = indexname , encoding = "utf-8")
    df = pd.DataFrame(raw_data)
    val_arr = df.values
    
    if filt :
        
        mean = np.mean(val_arr)
        scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
        val_arr_scal = scaler.fit_transform(val_arr)
        train_arr = val_arr_scal
        clf = svm.OneClassSVM(nu = 0.003, kernel = 'rbf' , gamma = 'auto')
        clf.fit(train_arr)
        pre = clf.predict(train_arr)
        ano_p_index = np.where(pre == -1)[0]
        
        for i in ano_p_index:

            fil_arr = val_arr
            fil_arr[i] = random.randint(int(mean*0.99),int(mean*1.01))
                
    else :
        
        fil_arr = val_arr

    return fil_arr

def resample_data(data , input_len , daily_ts):

    print("resampling data...")

    data_buffer = np.array([])
    sequence_length = input_len + daily_ts
    fil_arr = np.reshape(data , (data.shape[0],))
    #print(fil_arr.shape)

    for pro_time in range(int((len(fil_arr)/daily_ts) - (sequence_length/daily_ts))):

        data_buffer = np.append(data_buffer,fil_arr[(pro_time*daily_ts):(pro_time*daily_ts)+sequence_length])

    data_buffer = np.reshape(data_buffer,(int((len(fil_arr)/daily_ts) - (sequence_length/daily_ts)),sequence_length))

    #print(data_buffer,data_buffer.shape)
    process_data = data_buffer[:,:input_len]
    target_dataset = data_buffer[:,-daily_ts:]

    print("==========================================")
    print("shape of process data" , process_data.shape)
    print("shape of target dataset" , target_dataset.shape)
    print("==========================================")

    return process_data , target_dataset



