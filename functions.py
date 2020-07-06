import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import collections
import numpy as np
import pickle
import re
from imblearn.combine import SMOTETomek 

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
  
#import XGBoost classifier and accuracy
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def logistic_reg(data,y):
    # X - data frame que possui as varíveis de input no sistema
    # Y - valores a serem acertados no modelo
    X = data.drop(columns=y)
    Y = data[y]

    # separa grupo de teste e treino
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)
    
    
    #x_train, y_train = over_sampling(x_train, y_train)

    # treina regressão logistica
    logisticRegr = LogisticRegression(solver='newton-cg',max_iter=1000)
    logisticRegr.fit(x_train, y_train)

    # realiza redição e obtem score
    predictions = logisticRegr.predict(x_test)
    score = logisticRegr.score(x_test, y_test)

    # Save the model as a pickle in a file 
    pickle.dump(logisticRegr, open("logreg_01.pickle", 'wb'))

    resultado(y_test,predictions)

    return   

def xg_boost(data,y):

    # X - data frame que possui as varíveis de input no sistema
    # Y - valores a serem acertados no modelo
    X = data.drop(columns=y)
    Y = data[y]

    # separa grupo de teste e treino
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)
    
    #x_train, y_train = over_sampling(x_train, y_train)

    #instantiate model and train
    model = XGBClassifier(learning_rate = 0.01, n_estimators=1000, max_depth=8,scale_pos_weight=1.1)
    model.fit(x_train, y_train)

    # make predictions for test set
    y_pred = model.predict(x_test)
    predictions = [round(value) for value in y_pred]

    
    # Save the model as a pickle in a file 
    pickle.dump(model, open("xgboost_01.pickle", 'wb'))

    
    resultado(y_test,predictions)
    
    return                

def resultado(y_test,predictions):
        
    # quantidade de 0 e 1 predicted 
    collections.Counter(predictions)
    num_0 = collections.Counter(predictions)[0] 
    num_1 = collections.Counter(predictions)[1] 
    soma = num_0 + num_1 
    
    print("Numero de 0: {0}, {1:.1%} \nNumero de 1: {2}, {3:.1%}\n".format(num_0,num_0/soma,num_1,num_1/soma))
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    
    results = pd.DataFrame()
    results['y_test'] = y_test
    results['predict'] = predictions
    results.loc[(results['y_test'] == results['predict']) & (results['y_test'] == 0), 'Resultado'] = 'Correto Negativo'
    results.loc[(results['y_test'] == results['predict']) & (results['y_test'] == 1), 'Resultado'] = 'Correto Positivo'
    results.loc[(results['y_test'] != results['predict']) & (results['y_test'] == 0), 'Resultado'] = 'Falso Positivo'
    results.loc[(results['y_test'] != results['predict']) & (results['y_test'] == 1), 'Resultado'] = 'Falso Negativo'
    print(results.groupby(['Resultado'])['predict'].count())

    _ = results.groupby(['Resultado'])['predict'].count().values
    

    print('\nTaxa de Acerto Negativos:{0:.1%} \nTaxa de Acerto Positivos: {1:.1%}'.format(_[0]/(_[0] + _[3]),_[1] /(_[1]+ _[2])))

    return



def process_data_rs(data_rs):
    
    data_rs = data_rs.drop(columns=['COD_IBGE','REGIAO_COVID','COD_REGIAO_COVID','DATA_CONFIRMACAO','DATA_SINTOMAS','DATA_EVOLUCAO','RACA_COR','DATA_EVOLUCAO_ESTIMADA',
                                    'DATA_INCLUSAO_OBITO'])

    columns = ('HOSPITALIZACAO','FEBRE','TOSSE','GARGANTA','DISPNEIA','OUTROS')
    for x in columns:
        data_rs[x] = data_rs[x].map(dict(sim=1, nao=0,SIM=1,NAO=0,Sim=1,Nao=0))



    #trabalho com colunas que possuem valores em string
    columns_dummies = ['MUNICIPIO','SEXO','FAIXAETARIA','CRITERIO']
    df_dummies = pd.DataFrame()
    for x in columns_dummies:
        df_dummies = pd.concat([df_dummies,pd.get_dummies(data_rs[x])],axis=1)
    data_rs = pd.concat([data_rs,df_dummies],axis=1)


    #morbidades dummies
    morb_dummies = data_rs['COMORBIDADES'].str.get_dummies(sep=',')

    data_rs = pd.concat([data_rs,morb_dummies],axis=1)


    data_rs = data_rs.fillna(-9999)



    #drop columns que foram criados os dummies e variaveis não utilziadas
    columns_to_drop = columns_dummies + ["COMORBIDADES"]
    data_rs = data_rs.drop(columns=columns_to_drop)



    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    data_rs.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in data_rs.columns.values]

    
    
    
    return data_rs


def run_xgboost(x_run):
    model = pickle.load(open("Models/xgboost_09.pickle", 'rb'))
    
    y_pred = model.predict(x_run)
    predictions = [round(value) for value in y_pred]
    
    return predictions

def run_logreg(x_run):
    model = pickle.load(open("Models/logreg_02.pickle", 'rb'))
    
    y_pred = model.predict(x_run)
    predictions = [round(value) for value in y_pred]
    
    return predictions


def over_sampling(x,y):
    
    smotomek = SMOTETomek(random_state=1)
    bal_x, bal_y= smotomek.fit_resample(x, y)
    
    
    return bal_x,bal_y


def run_deepnn(x_run):

    model = keras.models.load_model('Models/deep_neural_network')

    
    y_pred = model.predict(x_run)
    y_pred =(y_pred>0.5).astype(int)[:,0]
    predictions = [round(value) for value in y_pred]
    
    return predictions



def run_convnet(x_run):

    model = keras.models.load_model('Models/conv_net_02')
    x_run= x_run.to_numpy()
    x_run = x_run.reshape(x_run.shape[0],478,1,1)

    
    y_pred = model.predict(x_run)
    y_pred =(y_pred>0.5).astype(int)[:,0]
    predictions = [round(value) for value in y_pred]
    
    return predictions