import numpy as np
import pandas as pd
import configparser

config = configparser.ConfigParser()
config.read('myconfig.ini')
initial_capital = config["INVESTMENT"].getfloat('initial_capital')
transaction_cost = config["INVESTMENT"].getfloat('transaction_cost')
asset_number = config["INVESTMENT"].getint('asset_number') 

def loadData(path):
    data = pd.read_csv(path)[:-1]
    data.dropna(inplace =True)
    split_point = int(len(data)*config["NEURAL_NET"].getfloat('train_size')) 
    close = data["Close"].iloc[split_point:]
    high = data["High"].iloc[split_point:]
    low = data["Low"].iloc[split_point:]
    return close, high, low

def calculate_transaction(portfolio):
    portfolio['cash']=initial_capital-np.array([(a*b+(a*b*transaction_cost)) for a,b in zip(portfolio['pos_diff'],portfolio['close'])]).cumsum()
    return portfolio
    
def tradeStrategy(pred_classes, path, dates):
    print("Creating investment portfolio....")
    close,high,low=loadData(path)
    print(len(dates))
    dates = dates[:-1]
    dates = dates.iloc[int(len(dates)-len(close)):]
    dates.reset_index(drop=True ,inplace=True)
    close.reset_index(drop=True, inplace=True)
    portfolio = pd.DataFrame(dates)
    portfolio['close']=close
    portfolio['signal'] = pred_classes 
    portfolio.set_value(len(portfolio)-1,'signal',0)
    print('pred_classes: '+str(len(pred_classes)))
    print('dates: '+str(len(dates)))
    print('close: '+str(len(close)))
    #gives an array with 1 multiplied by 90000 
    portfolio['positions'] = portfolio['signal']*asset_number 
    #multiplies the number of assets by the close price
    portfolio['positions_value'] = np.multiply(portfolio['close'].values, portfolio['positions'].values)
    #calculates the difference between t+1 and t
    #makes the first entry 0 since it is NaN
    portfolio['pos_diff'] = portfolio['positions'].diff()
    portfolio.set_value(0,'pos_diff',(portfolio['positions'].get(0))*(portfolio['signal'].get(0)))
    #portfolio['cash'] = initial_capital-np.multiply(portfolio['pos_diff'], portfolio['close']).cumsum()
    portfolio=calculate_transaction(portfolio)
    #portfolio['pos_diff'][0] = portfolio['positions'].get(0)
    #portfolio.set_value(0,'pos_diff',portfolio['positions'].get(0))
    # Add `total` to portfolio
    portfolio['total'] = portfolio['cash'] + portfolio['positions_value']
    # Add `returns` to portfolio
    portfolio['returns'] = portfolio['total'].pct_change()
    portfolio.to_csv("datasets/final_portfolio.csv", index=False)
    print("Created investment portfolio!")
    print("\n"+"------------------Portfolio Stats-------------------"+"\n")
    roi = ((portfolio['total'].iloc[-1]-initial_capital)/initial_capital)*100
    print("ROI: "+str(roi)+"%")
    number_of_transactions = len([n for n in portfolio['pos_diff'] if n!=0])
    file = open("datasets/final_portfolio.csv","r")
    text=file.read()
    file.close()
    file = open("datasets/final_portfolio.csv","w")
    file.write("\n")
    file.write("------------------PORTFOLIO STATS--------------------")
    file.write("\n")
    file.write("ROI,"+str(roi)+" %"+"\n")
    file.write("Number of transactions,"+str(number_of_transactions)+"\n")
    print("Number of transactions: "+str(number_of_transactions))
    max_dd = ((portfolio['total'].max()-portfolio['total'].min())/portfolio['total'].max())*100
    file.write("Max portfolio value, "+str(portfolio['total'].max())+"\n")
    file.write("Min portfolio value, "+str(portfolio['total'].min())+"\n")
    file.write("Max Drawdown:, "+str(max_dd)+" %"+"\n")
    print("Max portfolio value: "+str(portfolio['total'].max()))
    print("Min portfolio value: "+str(portfolio['total'].min()))
    print("Max Drawdown: "+str(max_dd)+" %")
    file.write("\n")
    file.write(text)
    file.close()