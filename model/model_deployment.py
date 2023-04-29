#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os

def predict(year, mileage, state, make, model):
    
    # Carga de datos de archivo .csv
    dataTraining = pd.read_csv('https://github.com/elkindavid/MIAD_UNIANDES/blob/main/dataTrain_carListings.csv', error_bad_lines=False)
    data = dataTraining

    # Codificación de las variables categoricas
    cat = ['State','Make','Model']
    dic = {'State':{},'Make':{},'Model':{}}

    for i in cat:
        idx, codex = pd.factorize(data[i])
        data[i] = idx
        # Diccionario de referencia
        dic[i].update({code: i for i, code in enumerate(codex)})
    
    reg = joblib.load(os.path.dirname(__file__) + '/car_price_reg.pkl') 
    
    state_ = dic['State'][state]
    make_ = dic['Make'][make]
    model_ = dic['Model'][model]
    
    car_ = pd.DataFrame([[year, mileage, state_, make_, model_]], columns=['Year','Mileage','State','Make','Model'])   
    car_['YxM'] = (year * mileage)
    
#     Make prediction
    p1 = reg.predict(car_)

    return p1


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add an URL')
    else:
        
        year = sys.argv[1]
        mileage = sys.argv[2]
        state = sys.argv[3]
        make = sys.argv[4]
        model = sys.argv[5]
        
        p1 = predict(year, mileage, state, make, model)
        
        print(car)
        print('Car Price: ', p1)
        