import os
import sys
import warnings
import time

import tensorflow as tf

from keras.layers import Dense 
from keras.models import Sequential

MODEL_PATH = "C:/Users/108319004/Desktop/pyworkspace/AUO_forecasting/project_AUO/model_repo/"

def build_model(Tr_X , Tr_Y , Te_X , Te_Y , Batch_s , Epoc):

    print("Training start")

    model = Sequential()
    model.add(
        Dense(int(2*(Tr_X.shape[1])),
        input_shape = (Tr_X.shape[1],),
        activation = 'relu'))
    model.add(Dense(int(2*(Tr_X.shape[1])),
        activation = 'sigmoid'))
    model.add(Dense(int(2*(Tr_X.shape[1])),
        activation = 'sigmoid'))
    model.add(Dense(int(4*(Tr_X.shape[1])),
        activation = 'sigmoid'))
    model.add(Dense(int(6*(Tr_X.shape[1])),
        activation = 'sigmoid'))
    model.add(Dense(int(6*(Tr_X.shape[1])),
        activation = 'sigmoid'))
    model.add(Dense(int(4*(Tr_X.shape[1])),
        activation = 'sigmoid'))
    model.add(Dense(int(2*(Tr_X.shape[1])),
        activation = 'sigmoid'))
    model.add(Dense(int(2*(Tr_X.shape[1])),
        activation = 'sigmoid'))
    model.add(Dense(1,activation='linear'))

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    warnings.filterwarnings("ignore") 
    start = time.time()
    model.compile(optimizer="adam", loss="mae")
    print("> Compilation Time : ", time.time() - start)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1 ,patience=150)
    mc = ModelCheckpoint(MODEL_PATH+'model_sample.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    #model.fit(Tr_X,Tr_Y,batch_size=Batch_s,nb_epoch=Epoc,validation_split=0.05)
    model.fit(Tr_X,Tr_Y,batch_size=Batch_s,nb_epoch=Epoc,validation_split=0.05,callbacks=[es , mc])
    #model.save(MODEL_PATH+'model_sample.h5')
    #save_model = load_model(MODEL_PATH+'model_sample.h5')

    return model