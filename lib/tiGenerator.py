import pandas as pd
import numpy as np
from pyti.exponential_moving_average import exponential_moving_average as ema
from pyti.simple_moving_average import simple_moving_average as sma
from pyti.relative_strength_index import relative_strength_index as rsi
from pyti.momentum import momentum as mom
from pyti.average_true_range import average_true_range as atr
from pyti.average_true_range_percent import average_true_range_percent as atrp
from pyti.bollinger_bands import upper_bollinger_band as up_bb
from pyti.bollinger_bands import middle_bollinger_band as mid_bb
from pyti.bollinger_bands import percent_b as per_bb
from pyti.bollinger_bands import lower_bollinger_band as low_bb
from pyti.standard_deviation import standard_deviation as std
from pyti.standard_variance import standard_variance as stv
from pyti.detrended_price_oscillator import detrended_price_oscillator as dpo
from pyti.aroon import aroon_up as aa_up
from pyti.aroon import aroon_down as aa_down
from pyti.chande_momentum_oscillator import chande_momentum_oscillator as cmo
from pyti.double_exponential_moving_average import double_exponential_moving_average as dema
from pyti.double_smoothed_stochastic import double_smoothed_stochastic as dss
from pyti.hull_moving_average import hull_moving_average as hma
from pyti.rate_of_change import rate_of_change as roc
from pathlib import Path

def kurt(data, period):
    values = pd.Series(data).pct_change()
    data = values.rolling(period).kurt()
    return data
    
def skew(data, period):
    values = pd.Series(data).pct_change()
    data = values.rolling(period).skew()
    return data

def createCsv(close, lower_bound, upper_bound, ti):
    csv = pd.DataFrame()
    for i in range (lower_bound, upper_bound+1):
        csv[str(i)] = ti(close, i)  
    return csv

def tiGenerator(close, ti, lower_bound, upper_bound):
    path='ti/'
    normal_ti=np.array([])
    special_ti=np.array([])
    function_mappings = {
    'ema': ema, 'sma':sma, 'hma':hma, 'aa_down':aa_down, 'aa_up':aa_up, 'mom':mom, 'roc':roc,  
    'rsi':rsi,'low_bb':low_bb, 'mid_bb':mid_bb, 'per_bb':per_bb, 'up_bb':up_bb, 'cmo':cmo, 
    'dss':dss, 'atr':atr, 'atrp':atrp, 'dpo':dpo, 'kurt':kurt, 'skew':skew, 
    'std':std, 'stv':stv, 'dema':dema
    }       
    for t in ti:
        file = Path(path+str(t)+".csv")
        if file.exists():
            print('Normal ti: '+str(t)+".csv already exists" )
            normal_ti=np.append(normal_ti,t)
        else:
            if t in function_mappings:
                print('Normal ti: '+str(t)+".csv does not exist. It will be created." )
                d = createCsv(close, lower_bound, upper_bound, function_mappings[t])    
                d.to_csv(path+str(t)+".csv", index=False)
                normal_ti=np.append(normal_ti,t)
            else:
                print("Special ti: "+t+" encountred. It will be calculated on the fly!")
                special_ti=np.append(special_ti,t)
    return normal_ti, special_ti

    