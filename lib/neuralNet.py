import numpy as np
import pandas as pd
import random
import os
random.seed(32)
np.random.seed(32)
os.environ["PYTHONHASHSEED"] = "0"
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.backend.tensorflow_backend
from keras import backend as K
import configparser

def neural_net(neurons1, neurons2, X_train, X_test, y_train, y_test, mode=0):
    config = configparser.ConfigParser()
    config.read('myconfig.ini')
    split_point = len(X_train)-int(len(X_train)*config['NEURAL_NET'].getfloat('validation_size'))
    X_val = X_train[split_point:]
    X_train = X_train[:split_point]
    y_val = y_train[split_point:] 
    y_train = y_train[:split_point]
    
    model = Sequential()
    model.add(Dense(neurons1, input_dim=X_train.shape[1], activation=config['NEURAL_NET']['activation']))
    #model.add(BatchNormalization())
    model.add(Dense(neurons2, activation=config['NEURAL_NET']['activation']))
    model.add(Dense(1, activation='sigmoid'))
    filepath="BESTMODEL.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_acc', patience=20, verbose=1, min_delta=0.001) 
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=config['NEURAL_NET']['optimizer'], metrics=['accuracy'])
    model_history = model.fit(X_train, y_train,  validation_data=[X_val, y_val],
                                 shuffle=False, epochs=config['NEURAL_NET'].getint('epochs')
                                , batch_size=config['NEURAL_NET'].getint('batch_size') 
                                , verbose=1, callbacks=[early_stop,checkpoint])
    model.load_weights(filepath)
    pred_classes_test = model.predict_classes(X_test).flatten().tolist()
    pred_classes_val = model.predict_classes(X_val).flatten().tolist()
    eval_test = model.evaluate(X_test, y_test)
    eval_val = model.evaluate(X_val, y_val)
    print("\nValidation evaluation: "+str(eval_val))
    print("\nTest evaluation"+str(eval_test))
    clean_session()
    K.clear_session()
    return pred_classes_test, pred_classes_val, eval_test, eval_val, model_history

def clean_session(): 
    if keras.backend.tensorflow_backend._SESSION:
       import tensorflow as tf
       tf.reset_default_graph() 
       keras.backend.tensorflow_backend._SESSION.close()
       keras.backend.tensorflow_backend._SESSION = None
