import lib.featureAddition as fa
import lib.featureMap as fm
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random
import lib.preprocessing as pp
import lib.neuralNet as nn
random.seed(32)
import numpy as np
import configparser
from sklearn.metrics import confusion_matrix
import lib.investment2 as im

class GA(object):
    file=0
    data=[]
    special_ti = []
    normal_ti = []
    pred_classes_test=[]
    pred_classes_val=[]
    last_accuracy=0
    best_eval = 0
    model_history=0
    config=0
    best_y_test=[]
    evaluation_val = []
    evaluation_test = []
    fitness_func = ""
    roi = 0
    last_roi=-100 # Pedraria gigante so para passar a primeira iteracao e a comparacao com zero qdo for ROI
                  # negativo passar
        
    
    def __init__(self, data, normal_ti, special_ti):
        self.data=data
        self.normal_ti=normal_ti
        self.special_ti=special_ti
        
    def ga_run(self):
        self.file = open("history.txt","w")
        self.config = configparser.ConfigParser()
        self.config.read('myconfig.ini')
        pred_classes= self.ga(self.file, self.data, self.normal_ti, self.special_ti)
        return pred_classes

    def evalFitnessMax(self, individual):
        print ("")
        print ("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print ("Chrmossome structure:")
        print (individual)
        fitness=self.process(individual, self.data, self.normal_ti, 
                         self.special_ti, mode=0)
        
        return fitness,

    def ga(self, file, data, normal_ti, special_ti):
        self.file = open("history.txt","a")
        features_size = fm.getIndiceArraySize(np.append(normal_ti,special_ti))
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        upper_bound=self.config['TI_FEATURES'].getint('upper_bound')
        lower_bound=self.config['TI_FEATURES'].getint('lower_bound')
        toolbox.register("attr_bool", random.randint, lower_bound, upper_bound)
        toolbox.register("individual", tools.initRepeat, creator.Individual, 
        toolbox.attr_bool, (features_size*2)+4) # +4 equals t the number of neurons needed to be add  in oreder to control the number of neurons in the net
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evalFitnessMax)
        toolbox.register("mate", self.customCrossover)
        toolbox.register("mutate", self.customMutation)
        toolbox.register("select", tools.selTournament, tournsize=self.config['GA'].getint('tournsize'))
        pop = toolbox.population(self.config['GA'].getint('pop')) # initial population
        # number of genrations, corssover and mutation probabilities
        ngen = self.config['GA'].getint('ngen')
        cxpb = self.config['GA'].getfloat('cxpb')
        mutpb = self.config['GA'].getfloat('mutpb')
        self.fitness_func =self.config['GA']['fitness_func']
        print("Fitness function: "+str(self.fitness_func))
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        counter=1
        for ind, fit in zip(pop, fitnesses):
            self.file.write("\nIndividual number: "+str(counter))
            self.file.write("\n"+str(ind)+"\n")
            print("\nIndividual number: "+str(counter))
            ind.fitness.values = fit
            print("Fitness value: "+str(fit)+" | "+ "Validation Acc: "+str(self.evaluation_val[1])+" | "+ "Test Acc: "+str(self.evaluation_test[1])+" | "+ "ROI: "+str(self.roi)+'\n')
            self.file.write("Fitness value: "+str(fit)+" | "+ "Validation Acc: "+str(self.evaluation_val[1])+" | "+ "Test Acc: "+str(self.evaluation_test[1])+" | "+ "ROI: "+str(self.roi)+'\n')
            counter=counter+1
        for g in range(ngen):
            print ("\n"+"Generation: "+str(g))
            self.file.write("\ngeneration: "+str(g)+"\n")
            print("population length: "+str(len(pop)))
            pop = toolbox.select(pop, k=len(pop))
            pop = algorithms.varAnd(pop, toolbox, cxpb, mutpb)
            invalids = [ind for ind in pop if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalids)
            count = 0
            for ind, fit in zip(invalids, fitnesses):
                count = count+1
                print("\nIndividual number: "+str(count)+" | "+"Generation: "+str(g))
                ind.fitness.values = fit
                self.file.write("\nIndividual number: "+str(count)+" | "+"Generation: "+str(g)+"\n")
                self.file.write("\n"+str(ind)+"\n")
                self.file.write("Fitness value: "+str(fit)+" | "+ "Validation Acc: "+str(self.evaluation_val[1])+" | "+ "Test Acc: "+str(self.evaluation_test[1])+'\n')
                print("Fitness value: "+str(fit)+" | "+ "Validation Acc: "+str(self.evaluation_val[1])+" | "+ "Test Acc: "+str(self.evaluation_test[1])+" | "+ "ROI: "+str(self.roi)+'\n')
        print("\n+++++++++++++++++++++++++++++++ BEST CHROMOSOME ++++++++++++++++++++++++++++++++\n ")
        self.file.write("\n"+"Best element"+"\n")
        best_run = tools.selBest(pop, k=1)[0]
        self.file.write(str(best_run)+"\n")
        self.file.write(self.fitness_func+" of best element before last run: "+str(best_run.fitness.values[0])+"\n")
        print("Fitness value: "+str(best_run.fitness.values[0]))
        print("Optimization "+self.config["GA"]["optimization_set"]+" loss: "+str(self.best_eval[0]))
        print("Optimization "+self.config["GA"]["optimization_set"]+" accuracy: "+str(self.best_eval[1])+"\n")
        print("Optimization "+self.config["GA"]["optimization_set"]+ " ROI: "+str(self.last_roi)+"\n")
        self.confusionMatrix(self.best_y_test, self.pred_classes_test)
        self.file.write("Optimization "+self.config["GA"]["optimization_set"]+ " ROI: "+str(self.last_roi)+"\n")
        self.file.write("Optimization "+self.config["GA"]["optimization_set"]+" accuracy: "+str(self.best_eval[1])+"\n")
        self.file.write("Pred classes:\n"+str(self.pred_classes_test))
        return self.pred_classes_test
    
    def process(self, indices, data, normal_ti, special_ti, mode):
        process_data=data.copy()
        print("")
        print("--------------------------- FEATURE ADDITION -----------------------------")
        print("")
        print("Normal_ti")
        print(normal_ti)
        print("Special ti")
        print(special_ti)
        final_data, net_ind = fa.addFeatures(normal_ti, special_ti, process_data, indices)
        print("-------------------------------------------------------------------------")
        print("")
        print("Data columns before addition: "+str(len(data.columns)))
        print(data.columns)
        print("Data columns after addition: "+str(len(final_data.columns)))
        print(final_data.columns)
        print("neural_net indices: ")
        print(net_ind)
        X_train, X_test, y_train, y_test, scalers = pp.preprocessData(final_data) 
        print(len(y_test))
        print("\nData length after addition:\n" +"Train: "+str(len(X_train))+"\nTest: "+str(len(X_test)))
        print("\nDropped "+str(len(data)-len(X_train)-len(X_test))+" rows from original data")
        print("")
        print("-------------------------------------------------------------------------")
        print("")
        pred_test, pred_val, self.evaluation_test, self.evaluation_val, model_history = nn.neural_net(net_ind[0], net_ind[1], X_train, X_test, y_train, y_test)
        print("\n-----------------------------NEURAL NET SCORE---------------------------------------")
        print("")
        print("Validation accuracy; "+str(self.evaluation_val[1]))
        print("Test accuracy: "+str(self.evaluation_test[1]))
        optimization_set = self.config["GA"]['optimization_set']
        print("\n-----------------------------OPTIMIZATION PROCESS---------------------------------------")
        print("\nOptimization will be performed on the {} set".format(optimization_set))
        if optimization_set=="val":
            evaluation = self.evaluation_val
            roi = im.tradeStrategy(pred_val, mode=2)
        elif optimization_set=="test":
            evaluation = self.evaluation_test
            roi = im.tradeStrategy(pred_test, mode=2)        
        if self.fitness_func == "accuracy":
            last_metric = self.last_accuracy
            metric = evaluation[1]
        elif self.fitness_func == "roi":
            last_metric = self.last_roi
            metric = roi 
        self.roi = roi
        if metric > last_metric:
            print("\nAccuracy improved! Saving prediction, best element till now.")
            self.last_roi = roi
            self.pred_classes_test=pred_test
            self.pred_classes_val=pred_val
            self.last_accuracy=evaluation[1]
            self.best_eval=evaluation
            self.best_y_test = y_test
        return metric

    def customCrossover(self, ind1, ind2):
        split = int(len(ind1)/2)
        half_ind1_values = ind1[:split]
        half_ind1_presence = ind1[split:]
        half_ind2_values = ind2[:split]
        half_ind2_presence = ind2[split:]
        final_ind_values1, final_ind_values2 = tools.cxTwoPoint(half_ind1_values, half_ind2_values)
        final_ind_presence1, final_ind_presence2 =tools.cxTwoPoint(half_ind1_presence, half_ind2_presence)
        ind1[:split], ind1[split:] = final_ind_values1, final_ind_presence1
        ind2[:split], ind2[split:] = final_ind_values2, final_ind_presence2
        return ind1, ind2
    
    def customMutation(self, ind, low=5, up=100, indpb=0.05):
        split = int(len(ind)/2)
        half_ind_values = ind[:split]
        half_ind_presence = ind[split:]
        final_ind_values = tools.mutUniformInt(half_ind_values, low, up, indpb)
        final_ind_presences = tools.mutUniformInt(half_ind_presence, low, up, indpb)
        ind[:split], ind[split:] = final_ind_values[0], final_ind_presences[0]
        return ind,
    
    def confusionMatrix(self, y_test, pred):
        cm = confusion_matrix(y_test, pred)
        print(cm)
        self.file.write("Confusion matrix metrics+\n")
        self.file.write(str(cm)+'\n')
        print("Test Accuracy: "+str((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))+'\n')
        print("Test Precision: "+str(cm[0][0]/(cm[0][0]+cm[0][1]))+'\n')
        print("Test Recall: "+str(cm[0][0]/(cm[0][0]+cm[1][0]))+'\n')
        self.file.write("Test Accuracy: "+str((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))+'\n')
        self.file.write("Test Precision: "+str(cm[0][0]/(cm[0][0]+cm[0][1]))+'\n')
        self.file.write("Test Recall: "+str(cm[0][0]/(cm[0][0]+cm[1][0]))+'\n')