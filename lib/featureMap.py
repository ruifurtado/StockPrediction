feature_map={'ema': 1, 'sma':1, 'hma':1, 'aa_down':1, 'aa_up':1, 'mom':1, 'roc':1,  
             'rsi':1,'low_bb':1, 'mid_bb':1, 'per_bb':1, 'up_bb':1, 'cmo':1, 
             'dss':1, 'atr':1, 'atrp':1, 'dpo':1, 'kurt':1, 'skew':1, 
             'std':1, 'stv':1, 'dema':1, 'cci':2, 'macd':2, 'po':2, 'adx':2}

def getIndiceArraySize(ti):
    size=0
    for i in ti:
       size+=feature_map[i]
    return size

def getIndice(i):
    return feature_map[i]