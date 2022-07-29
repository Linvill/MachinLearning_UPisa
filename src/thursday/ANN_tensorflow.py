#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

img_dataset=keras.datasets.fashion_mnist

(X_training, y_training), (X_test, y_test) = img_dataset.load_data()


# scale bw images in the range 0-1 and create a validation dataset

X_validation, X_trainsmall = X_training[:10000]/255., X_training[10000:]/255.
y_validation, y_trainsmall = y_training[:10000], y_training[10000:]

labels=['Tshirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Boot']


model = keras.models.Sequential()

model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

train_hist=model.fit(X_trainsmall, y_trainsmall, epochs=30, validation_data=(X_validation, y_validation))


pd.DataFrame(train_hist.history).plot(figsize=(8,5))
plt.grid(True)
plt.ylim([0,1])
plt.show()

model.evaluate(X_test, y_test)


X_new = X_test[:10]

y_prob= model.predict(X_new)
print(y_prob)
y_pred= model.predict_classes(X_new)

for iy_pred,iy_prob,iy_true in zip(y_pred,y_prob,y_test):
    print(labels[iy_pred],iy_prob,labels[iy_true])




