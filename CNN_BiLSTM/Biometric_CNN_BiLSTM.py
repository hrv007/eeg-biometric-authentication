print('start')

import pandas as pd
import numpy as np
import scipy
from scipy.io import loadmat
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("Defining Data generator")

datagen = ImageDataGenerator(rescale=1./255, 
                             validation_split=0.4)
train_dataset = datagen.flow_from_directory('Mel Spectrograms', 
                                           target_size=(150, 150), 
                                           batch_size=64)

valid_dataset = datagen.flow_from_directory('Mel Spectrograms', 
                                           target_size=(150, 150), 
                                           batch_size=32, subset='validation')

test_dataset = datagen.flow_from_directory('Mel Spectrograms', 
                                          target_size=(150, 150), 
                                          batch_size=32, 
                                          subset='validation')

print("Defining Model")

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(32, 
                              kernel_size=(3,3), 
                              activation='relu', 
                              strides=(1,1), 
                              input_shape=(150,150, 3)))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.MaxPooling2D((4,4)))
model.add(keras.layers.Reshape((-1, 64)))

model.add(keras.layers.Bidirectional(keras.layers.LSTM(64, activation='tanh', return_sequences=True)))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(128, activation='tanh', return_sequences=True)))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(64, activation='tanh')))

model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(20, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(1e-3), metrics=['accuracy'])

print("Model Fitting")

checkpoint_cb = keras.callbacks.ModelCheckpoint('CNN_BiLSTM_Model.h5', save_best_only=True)

history = model.fit_generator(train_dataset, epochs=15, validation_data=valid_dataset, callbacks = [checkpoint_cb])

print(model.evaluate(test_dataset))
#model.save("CNN_model.h5")

print("Plotting Graph")

df_hist = pd.DataFrame(history.history)
pd.DataFrame(history.history).plot(figsize = (8,5))
df_hist.to_csv('History_CNN_BiLSTM.csv')
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.savefig("CNN_BiLSTM_graph.png")
plt.show()

print("The End")