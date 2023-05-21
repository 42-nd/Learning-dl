import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras

def mini_bath_GD(X,y,epochs,learning_rate = 0.01):
    n = X.shape[0]
    n_features = X.shape[1]

    w = np.ones(shape=(n_features))
    b = 0
    
    loss_list =[]
    epochs_list = []

    for i in range(epochs):
        random_slice = random.randint(0,n-4)

        slice_x = X[random_slice:random_slice+3]
        slice_y = y[random_slice:random_slice+3]

        y_pred = np.dot(w,slice_x.T) + b
        
        dw = -(2/len(slice_x))*np.mean(np.dot(slice_x.T,(slice_y-y_pred)))
        db = -(2/len(slice_y))*np.sum((slice_y-y_pred))
        
        w = w-learning_rate*dw
        b = b-learning_rate*db

        loss = np.mean(np.square(slice_y-y_pred))
        
        if i%10 == 0:
            loss_list.append(loss)
            epochs_list.append(i)

    return w,b,loss,loss_list,epochs_list
    
def predict (area,bedrooms,w,b):
    scaled_x = MinMaxScaler().transform([[area,bedrooms]])[0]

    scaled_price = w[0]*scaled_x[0] + w[1] * scaled_x[1] + b
    return MinMaxScaler().inverse_transform([[scaled_price]])[0]

if __name__ == "__main__":

    df = pd.read_csv("homeprices_banglore.csv")
    x = MinMaxScaler().fit_transform(df.drop(['price'],axis=1))
    y = MinMaxScaler().fit_transform(df.price.values.reshape(df.shape[0],1))
    w,b,loss,loss_list,epochs_list = mini_bath_GD(x,y.reshape(df.shape[0],1),10000)
    
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(epochs_list,loss_list)
    plt.show()


'''
def mini_batch_gradient_descent(X, y_true, epochs = 100, batch_size = 5, learning_rate = 0.01):
    
    number_of_features = X.shape[1]
    # numpy array with 1 row and columns equal to number of features. In 
    # our case number_of_features = 3 (area, bedroom and age)
    w = np.ones(shape=(number_of_features)) 
    b = 0
    total_samples = X.shape[0] # number of rows in X
    
    if batch_size > total_samples: # In this case mini batch becomes same as batch gradient descent
        batch_size = total_samples
        
    cost_list = []
    epoch_list = []
    
    num_batches = int(total_samples/batch_size)
    
    for i in range(epochs):    
        random_indices = np.random.permutation(total_samples)
        X_tmp = X[random_indices]
        y_tmp = y_true[random_indices]
        
        for j in range(0,total_samples,batch_size):
            Xj = X_tmp[j:j+batch_size]
            yj = y_tmp[j:j+batch_size]
            y_predicted = np.dot(w, Xj.T) + b
            
            w_grad = -(2/len(Xj))*np.mean(Xj.T.dot(yj-y_predicted))
            b_grad = -(2/len(Xj))*np.sum(yj-y_predicted)
            
            w = w - learning_rate * w_grad
            b = b - learning_rate * b_grad
                
            cost = np.mean(np.square(yj-y_predicted)) # MSE (Mean Squared Error)
        
        if i%10==0:
            cost_list.append(cost)
            epoch_list.append(i)
        
    return w, b, cost, cost_list, epoch_list
'''