import numpy as np
import pandas as pd
import configparser
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
config.read('myconfig.ini')
initial_capital = config["INVESTMENT"].getfloat('initial_capital')
transaction_cost = config["INVESTMENT"].getfloat('transaction_cost')
asset_number = config["INVESTMENT"].getint('asset_number') 
path = config['FILE']['path']

def create_order(action, cash, signal_order, capital, transactions=1):
    if action == "buy":
        current_cash = capital-(signal_order*transactions+(signal_order*transactions*transaction_cost))
    elif action == "sell":
        current_cash = capital+(signal_order*transactions-(signal_order*transactions*transaction_cost))
    return current_cash    
    
def create_cash(portfolio):
    cash = []
    total = []
    previous_position = 0          
    capital = initial_capital
    for idx, signal_order in enumerate(zip(portfolio['signal'],portfolio['positions_value'])):
        if idx==len(portfolio)-1:
            if portfolio['positions'].iloc[-1] < 0:
                capital = capital-(signal_order[1]+(signal_order[1]*transaction_cost))
            if portfolio['positions'].iloc[-1] > 0:
                capital = capital+(signal_order[1]-(signal_order[1]*transaction_cost))
            cash.append(capital)
            total.append(capital)
            portfolio.set_value(len(portfolio)-1,'positions',0)
            portfolio.set_value(len(portfolio)-1,'positions_value',0)
            break
        if signal_order[0] == 1:   # long
            if previous_position == 0: 
                capital = create_order('buy', cash, signal_order[1], capital)
                cash.append(capital)  
                total.append(capital+portfolio['positions_value'].iloc[idx])
            elif previous_position == -1:  
                capital = create_order('buy', cash, signal_order[1], capital, transactions=2)
                cash.append(capital)
                total.append(capital+portfolio['positions_value'].iloc[idx])
            previous_position = 1  
        if signal_order[0] == -1:  # short
            if previous_position == 0: 
                capital = create_order('sell', cash, signal_order[1], capital)
                cash.append(capital)
                total.append(capital-portfolio['positions_value'].iloc[idx])
            elif previous_position == 1:  
                capital = create_order('sell', cash, signal_order[1], capital, transactions=2)
                cash.append(capital)
                total.append(capital-portfolio['positions_value'].iloc[idx])
            previous_position = -1  
        if signal_order[0] == 0:   # hold
            cash.append(capital)
            if previous_position == 0 or previous_position == -1:
                total.append(capital-portfolio['positions_value'].iloc[idx])
            if previous_position == 1:
                total.append(capital+portfolio['positions_value'].iloc[idx])
            #previous_position = 0
    portfolio['cash'] = cash
    portfolio['total'] = total
    return portfolio

def plotter(portfolio):
    portfolio[['signal','close','cash','positions_value','total']].plot(subplots=True, figsize=(6, 6)
    , title=['signal','close','cash','positions_value','total'])
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, ylabel='Portfolio value in $')
    # Plot the equity curve in dollars
    portfolio['total'].plot(ax=ax1, lw=2.)
    ax1.plot(portfolio.loc[portfolio['signal'] == 1].index, portfolio['total'][portfolio['signal']  == 1],'^', markersize=10 )
    ax1.plot(portfolio.loc[portfolio['signal'] == -1].index, portfolio['total'][portfolio['signal'] == -1],'v', markersize=10 )
    ax1.legend(['Portfolio value','Long','Short'])
    
    fig2 = plt.figure()
    ax_1 = fig2.add_subplot(111, ylabel='close entries')
    portfolio['close'].plot(ax=ax_1, lw=2.)
    ax_1.plot(portfolio.loc[portfolio['signal'] == 1].index, portfolio['close'][portfolio['signal']  == 1],'^', markersize=10 )
    ax_1.plot(portfolio.loc[portfolio['signal'] == -1].index, portfolio['close'][portfolio['signal'] == -1],'v', markersize=10 )
    ax_1.legend(['Market index','Long','Short'])
    plt.show()

def loadData(prediction):
    data = pd.read_csv(path)[:-1]
    data.dropna(inplace =True)
    split_point = len(prediction)
    data=data.iloc[-split_point:]
    close = data["Close"].reset_index(drop=True)
    high = data["High"].reset_index(drop=True)
    low = data["Low"].reset_index(drop=True)
    dates = data["Date"].reset_index(drop=True)
    return close, high, low, dates

def tradeStrategy(prediction, mode=1):
    if mode==1:
        print("Creating investment portfolio....")
    close,high,low,dates = loadData(prediction)
    portfolio = pd.DataFrame(dates)
    portfolio['close']=close
    #print(len(portfolio))
    #print(len(prediction))
    portfolio['prediction'] = prediction
    prediction_diff = pd.Series(prediction).diff()
    prediction_diff.iloc[0]=prediction[0]
    prediction_diff = np.array(prediction_diff).astype(int)
    portfolio['signal'] = prediction_diff
    #portfolio.set_value(len(portfolio)-1,'signal',0)
    no_zeros=portfolio['signal'].replace(to_replace=0, method='ffill') # Take the zeros and remain with 1 and -1
    portfolio['positions'] = no_zeros*asset_number
    #portfolio.set_value(len(portfolio)-1,'positions',0)
    portfolio['positions_value']=np.abs(portfolio['positions']*portfolio['close'])    
    portfolio = create_cash(portfolio)
    # Add `returns` to portfolio
    portfolio['returns'] = portfolio['total'].pct_change()
    if mode == 1:
        print("\n"+"------------------Portfolio Stats-------------------"+"\n")
    roi = ((portfolio['total'].iloc[-1]-initial_capital)/initial_capital)*100
    print("ROI: "+str(roi)+"%")
    if mode!=1:
        print("ROI used as fitness function!")
        return roi 
    portfolio.to_csv("datasets/final_portfolio2.csv", index=False)
    print("Created investment portfolio!")
    plotter(portfolio)
    