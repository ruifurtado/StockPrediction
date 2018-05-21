import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import configparser

def create_labels(data):
    y = data['Close'].pct_change().shift(-1).values
    y_label = np.array([1 if close > 0 else 0 for close in y]) # take the last value that has a NaN
    return y_label

def create_X_train_test(data, y_label):
    data = data[:-1] # the last row can't be used because y has a NaN
    y_label = y_label[:-1]
    config=configparser.ConfigParser()
    config.read('myconfig.ini')
    split_point = int(len(data)*config['NEURAL_NET'].getfloat('train_size'))
    X_train_original = data.iloc[:split_point]
    X_test_original = data.iloc[split_point:]
    y_train = y_label[:split_point]
    y_test = y_label[split_point:]
    return X_train_original, X_test_original, y_train, y_test

def standarize_train_data(data):
    data_norm = []
    scaler_list = np.array([])    
    columns = data.columns
    for i in columns:
        sc= StandardScaler()
        feature= sc.fit_transform(data[i].values.reshape(-1,1)).flatten()
        data_norm.append(feature)
        scaler_list=np.append(scaler_list,sc)
    data_norm=np.array(data_norm).T
    return data_norm, scaler_list

def standarize_test_data(data, scaler_list):
    data_norm = []
    columns = data.columns
    for i,sc in zip(columns,scaler_list):
        feature= sc.transform(data[i].values.reshape(-1,1)).flatten()
        data_norm.append(feature)
    data_norm=np.array(data_norm).T
    return data_norm

def preprocessData(data):
    data.dropna(inplace=True)
    y_label=create_labels(data)
    X_train_original, X_test_original, y_train, y_test=create_X_train_test(data, y_label)
    X_train, scalers = standarize_train_data(X_train_original)
    X_test = standarize_test_data(X_test_original, scalers)
    return X_train, X_test, y_train, y_test, scalers 