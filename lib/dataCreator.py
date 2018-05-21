import pandas as pd
import numpy as np
import lib.tiGenerator as tg
import configparser

# ti_csv vai ter os csvs loaded em memoria. secalhar nao e boa ideia e deveria ser alaterado. 
# A alternativa sera ler sempre do ficheiro

def dataCreator(path, ti):
    data,dates = loadData(path)
    special_ti, normal_ti=calculateTi(data,ti)
    return data, dates, special_ti, normal_ti
    

def loadData(path):
    data = pd.read_csv(path)
    dates = pd.to_datetime(data['Date'])
    data = data.iloc[:,2:]
    return data, dates

def calculateTi(data, ti):
    print("")
    print("--------------------------- TI CREATION -----------------------------")
    print("")
    config = configparser.ConfigParser()
    config.read('myconfig.ini')
    upper_bound=config['TI_FEATURES'].getint('upper_bound')
    lower_bound=config['TI_FEATURES'].getint('lower_bound')
    print("Desired ti's: ")
    print(ti)
    print("")
    normal_ti, special_ti=tg.tiGenerator(data['Close'].values, ti ,lower_bound, upper_bound)
    print("---------------------------------------------------------------------")
    print("")
    return special_ti, normal_ti
