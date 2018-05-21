from pyti.moving_average_convergence_divergence import moving_average_convergence_divergence as macd
from pyti.exponential_moving_average import exponential_moving_average as ema
from pyti.price_oscillator import price_oscillator as po
from pyti.commodity_channel_index import commodity_channel_index as cci
from pyti.directional_indicators import average_directional_index as adx
import lib.featureMap as fm
import numpy as np

def calculate_ti(special_ti, special_ti_indices, special_ti_presence_indices, data):
    function_mappings={'macd':macd_func, 'po':po_func, 
                   'cci':cci_func, 'adx':adx_func}
    for t in special_ti:
        number_of_indices = fm.getIndice(t)
        indices=np.array([])
        presences=np.array([])
        print("TI: "+t)
        for i in range(number_of_indices):
            indices=np.append(indices, special_ti_indices[0])
            special_ti_indices = special_ti_indices[1:]
            presences=np.append(presences, special_ti_presence_indices[0])
            special_ti_presence_indices = special_ti_presence_indices[1:]
        indices = indices.astype(int)
        presences = presences.astype(int) 
        print("Indices: "+str(indices))
        print("Presences: "+str(presences))
        data=function_mappings[t](indices, presences, data)
    return data

def macd_func(indices, presences, data):
    presence = np.mean([presences])
    if presence > 52.5:
        if indices[0]>indices[1]: 
            indices[0],indices[1] = indices[1],indices[0]
        print("indices: "+str(indices[0])+"  "+str(indices[1]))
        values="("+str(indices[0])+","+str(indices[1])+")"
        data['macd'+values] = macd(data['Close'].values, indices[0], indices[1])
        data['macd_sign'] = ema(data['macd'+values].values,9)
        data['macd_hist'] = data['macd'+values].values-data['macd_sign'].values
    return data
        
def po_func( indices, presences, data):
    presence = np.mean([presences])
    if presence > 52.5:
        if indices[0]>indices[1]: 
            indices[0],indices[1] = indices[1],indices[0]
        print("indices: "+str(indices[0])+"  "+str(indices[1]))
        data['po'+"("+str(indices[0])+","+str(indices[1])+")"] = po(data['Close'].values, indices[0], indices[1])
    return data
    
def cci_func(indices, presences, data):
    presence = np.mean([presences])
    if presence > 52.5:
        print("indices: "+str(indices[0]))
        data["cci"+str(indices[0])]=cci(data["Close"], data["High"], data['Low'], indices[0])
    return data
    
def adx_func(indices, presences, data):
    presence = np.mean([presences])
    if presence > 52.5:
        print("indices: "+str(indices[0]))
        data["adx"+str(indices[0])]=adx(data["Close"], data["High"],data['Low'], indices[0])
    return data
