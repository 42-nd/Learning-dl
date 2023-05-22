import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
def plot(X,y,index):
    plt.figure()
    plt.imshow(X[index])

(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train / 255
X_test = X_test / 255

X_train = X_train.reshape(-1,28,28,1)
cnn = keras.Sequential([
    keras.layers.Conv2D(10, (4,4), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D((2,2)),
 
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])
cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn.fit(X_train,y_train,epochs=2)

y_pred = cnn.predict(X_test)
y_pred = [np.argmax(elem) for elem in y_pred]
print(classification_report(y_pred,y_test))
