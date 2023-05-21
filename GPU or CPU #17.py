#%%timeit -n1 -r1
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
def get_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(1000,activation = 'relu'),
        keras.layers.Dense(500,activation = 'relu'),
        keras.layers.Dense(250,activation = 'relu'),
        keras.layers.Dense(125,activation = 'relu'),
        keras.layers.Dense(100,activation = 'relu'),
        keras.layers.Dense(10,activation = 'sigmoid')
    ])

    model.compile(optimizer='adam',
                    loss = 'sparse_categorical_crossentropy',
                    metrics = ['accuracy']
                  )
    return model
if __name__ == "__main__":

    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images_scaled = train_images/255
    train_labels_scaled = train_labels/255

    
    with tf.device('/CPU:0'):
        cpu_model = get_model()
        cpu_model.fit(train_images_scaled,train_labels_scaled,epochs = 1)
   

