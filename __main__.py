import lib.dataCreator as dc
from lib.ga import GA
import lib.investment2 as im
import configparser

def run():
    config = configparser.ConfigParser()
    config.read('myconfig.ini')
    path = config['FILE']['path']
    ti = [a for a in config['TI_FEATURES'] if config['TI_FEATURES'][a]=='True']
    data, dates, special_ti, normal_ti = dc.dataCreator(path,ti)
    ga = GA(data, normal_ti, special_ti)
    pred_classes = ga.ga_run()
    #final_data_path = "datasets/final_data.csv"
    im.tradeStrategy(pred_classes)
    
if __name__ == "__main__":
    run()