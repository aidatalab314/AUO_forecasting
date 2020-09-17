import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyEMD import CEEMDAN , Visualisation



def get_IMFset(data):
    
    print("start ensemble decompose")

    imfs_set = []
    ceemdan = CEEMDAN()

    for decon_time in range(data.shape[0]):

        series = np.reshape(data[decon_time],(data.shape[1],))
        imfs = ceemdan(series)
        imf, res = imfs[:-1], imfs[-1]
        imfs_set.append(imfs)

        print("processing No.",decon_time,"series")

    vis = Visualisation()
    vis.plot_imfs(imfs=imf, residue=res, include_residue=True)
    vis.show()

    return imfs_set

def excute_ae(list):

    return 0