import lib.featureMap as fm
import lib.specialTiGenerator as stg
import pandas as pd
import numpy as np

    
def addFeatures(normal_ti, special_ti, data, indices):
    net_neurons_indices = np.array([])
    net_neurons_presences = np.array([])
    indices = checkIndices(indices)
    splitter=int(len(indices)/2)
    ti_indices=indices[:splitter]
    net_neurons_indices=np.append(net_neurons_indices,ti_indices.pop())
    net_neurons_indices=np.append(net_neurons_indices,ti_indices.pop())
    presence_indices=indices[splitter:]
    net_neurons_presences=np.append(net_neurons_presences,presence_indices.pop())
    net_neurons_presences=np.append(net_neurons_presences,presence_indices.pop())
    print("")
    print("##################### Indices processing ###################\n")
    print("TI_indices: "+str(ti_indices))
    print("PRESENCE_indices: "+str(presence_indices))
    print("NORMAL TI LEN "+ str(len(normal_ti)))
    print(str(normal_ti))
    print("SPECIAL TI LEN "+ str(len(special_ti)))
    print(str(special_ti))
    print("")
    net_neurons_indices=net_neurons_indices.astype(int)
    data = splitTiPresences(data, normal_ti, special_ti, ti_indices, presence_indices)
    if net_neurons_indices[0]<len(data.columns): # neurons must be at least equal to features, cant be less
        net_neurons_indices[0]=len(data.columns)
    return data, net_neurons_indices


def checkIndices(indices):  # in order to avoid same TI's with the same value
    for i in range(0,len(indices)-1):   #minus 2 because we don't acces directly to last index
        if indices[i]==indices[i+1]:
            indices[i]=indices[i]-1
    return indices    
    
        
def splitTiPresences(data, normal_ti, special_ti, ti_indices, presence_indices):          # the last indices will always be given to the special TI's
    number_of_indices=fm.getIndiceArraySize(normal_ti)   # indices of normal TI
    normal_ti_indices=np.array(ti_indices[:number_of_indices])
    special_ti_indices=np.array(ti_indices[number_of_indices:])
    print("normal_ti_indices: "+str(normal_ti_indices))
    print("special_ti_indices: "+str(special_ti_indices))
    normal_ti_presence_indices=np.array(presence_indices[:number_of_indices])
    special_ti_presence_indices=np.array(presence_indices[number_of_indices:])
    print("normal_ti_presence_indices: "+str(normal_ti_presence_indices))
    print("special_ti_presence_indices: "+str(special_ti_presence_indices))
    print("")
    print("##################### Adding normal features ###################\n")
    for i in range(len(normal_ti)):
        ti_data = pd.read_csv('ti/'+normal_ti[i]+".csv")     # get indicator dataset
        if normal_ti_presence_indices[i] > 52.5:
            print("TI: "+str(normal_ti[i]))
            print("Indice: "+str(normal_ti_indices[i]))
            print("Presence: "+str(normal_ti_presence_indices[i]))
            column = ti_data.iloc[:,normal_ti_indices[i]-5]  
            data[str(normal_ti[i])+str(normal_ti_indices[i])]=column
    print("")
    print("##################### Adding special features ###################\n")
    data = stg.calculate_ti(special_ti, special_ti_indices, special_ti_presence_indices, data)
    print("")
    return data