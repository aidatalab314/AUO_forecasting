# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#ssa base filter
def getWindowMatrix(inputArray , t , m):

    temp = []
    n = t - m + 1

    for i in range(n):

        temp.append(inputArray[i:i+m])

    WindowMatrix = np.array(temp)

    return WindowMatrix


def SVDreduce (trajectoy_Matrix):

    u,s,v = np.linalg.svd(trajectoy_Matrix)
    m1,n1 = u.shape
    m2,n2 = v.shape
    index = s.argmax()
    u1 = u[:,index]
    v1 = v[index]
    u1 = u1.reshape((m1,1))
    v1 = v1.reshape((1,n2))
    value = s.max()
    newMatrix = value*(np.dot(u1,v1))

    return newMatrix


def recreateArray(newMatrix , t , m):

    ret = []
    n = t - m +1

    for p in range(1,t+1):

        if p<m:

            alpha = p

        elif p>t-m+1:

            alpha = t-p+1

        else:

            alpha = m

        sigma = 0

        for j in range(1,m+1):

            i = p - j +1

            if i>0 and i<n+1:

                sigma += newMatrix[i-1][j-1]

        ret.append(sigma/alpha)

    return ret


#get the multi-scale datastet
def get_ms_dataset(data , fil_seq_num , down_sam_num):

    print("start multi scale data process...")

    ms_data = []
    ms_data.append(data)

    print("start ssa base filtering and data down-sampling...")

    fil_buffer = np.array([])
    ds_buffer = np.array([])

    i = 0

    for item in data :

        i = i + 1

        for fil_t in range(fil_seq_num):

            fil_t = fil_t + 2
            filt_data = getWindowMatrix(item,len(item),fil_t)
            filt_data = SVDreduce (filt_data)
            filt_data = recreateArray(filt_data,len(item),fil_t)
            fil_buffer = np.append(fil_buffer , filt_data)

        for dow_t in range(down_sam_num):

            sub = item[dow_t::down_sam_num]
            ds_buffer = np.append(ds_buffer,sub)
        
        print("smoothing & down sampling for item" , i)

    print(fil_buffer.shape)
    print(ds_buffer.shape)

    fil_arr_set = np.reshape(fil_buffer , (data.shape[0] , fil_seq_num , data.shape[1]))
    ds_buffer_set = np.reshape(ds_buffer , (data.shape[0] , down_sam_num , int(data.shape[1]/down_sam_num)))
    ms_data.append(fil_arr_set)
    ms_data.append(ds_buffer_set)

    fig_fil , axes_fil = plt.subplots(fil_seq_num,1 ,figsize = (10,4))

    for time in range(fil_seq_num):

        ax = axes_fil[time]
        ax.plot(fil_arr_set[0,time,:], label='filter data')
        
    plt.legend()
    plt.show()

    fig_ds , axes_ds = plt.subplots(down_sam_num,1 ,figsize = (10,4))

    for time in range(down_sam_num):

        ax = axes_ds[time]
        ax.plot(ds_buffer_set[0,time,:], label='ds data')
        
    plt.legend()
    plt.show()

    return ms_data

