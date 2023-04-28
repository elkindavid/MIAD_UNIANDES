#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os

def predict(year, mileage):
    reg = joblib.load(os.path.dirname(__file__) + '/car_price_reg.pkl') 
    car_ = pd.DataFrame([[year, mileage]], columns=['Year','Mileage'])
    car_['Por_Make'] = 1
    car_['Por_State'] = 1
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
        
        p1 = predict(year, mileage)
        
        print(car)
        print('Car Price: ', p1)
        