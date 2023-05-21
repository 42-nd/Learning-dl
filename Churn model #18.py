import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report

import tensorflow as tf
from tensorflow import keras

if __name__ == "__main__":
    df = pd.read_csv('Churn_Modelling.csv',index_col='RowNumber')
    X = df[df.columns.tolist()[2:12]]
    y = df[df.columns.tolist()[-1:]]
    X['Gender'].replace({'Female':1,'Male':0},inplace=True)
    X = pd.get_dummies(data= X, columns=['Geography'])
    cols_to_scale = ['CreditScore','Age','Balance','Tenure','EstimatedSalary']
    X[cols_to_scale] = MinMaxScaler().fit_transform(X[cols_to_scale])
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.2,random_state=5)

    model = keras.Sequential([
        keras.layers.Dense(7,input_shape = (12,),activation='relu'),
        keras.layers.Dense(1,activation = 'sigmoid')
    ])
   
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train,y_train,epochs=10)

    yp = model.predict(X_test)
    y_pred = []
    for elem in yp:
        if elem > 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    print(classification_report(y_test,y_pred))