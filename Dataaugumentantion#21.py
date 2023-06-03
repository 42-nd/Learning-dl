import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url,  cache_dir='~', untar=True)

import pathlib
data_dir = pathlib.Path(data_dir)

flowers_labels_dict = {
    'roses': 0,
    'daisy': 1,
    'dandelion': 2,
    'sunflowers': 3,
    'tulips': 4,
}


X,y = [],[]
for flowers_names,class_img in flowers_labels_dict.items():
    images_paths = list(data_dir.glob(f'{flowers_names}/*.jpg'))
    for path in images_paths:
        img = cv2.imread(str(path))
        resize_img = cv2.resize(img,(180,180))
        X.append(resize_img)
        y.append(class_img)

X = np.array(X)
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
X_train = X_train/255
X_test = X_test/255
with tf.device('/GPU:0'):
    data_augmentation = keras.Sequential([
            keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(180,180,3)),
            keras.layers.experimental.preprocessing.RandomRotation(0.1),
            keras.layers.experimental.preprocessing.RandomZoom(0.1),
        ])
    model = keras.Sequential([
        data_augmentation,
        keras.layers.Conv2D(64,16,padding='same',activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(32,8,padding='same',activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(16,4,padding='same',activation='relu'),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(128,activation='relu'),
        keras.layers.Dense(5,activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    model.fit(X_train,y_train,epochs=10)
