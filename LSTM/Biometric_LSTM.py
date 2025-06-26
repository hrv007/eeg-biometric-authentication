print('start')

import pandas as pd
import numpy as np
import scipy
from scipy.io import loadmat
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split

print('File Loading')

df = pd.read_csv("Enrollment_Info.csv")

y = []
for subid in df['subject']:
    if subid == 'sub021':
        break
    y.append(subid)
y = np.array(pd.get_dummies(y))
print(y.shape, y)

x = []

for epochid in df['EpochID']:
    if epochid == 'epoch012228':
        break
    print(epochid)
    annots = loadmat('Enrollment/'+epochid+'.mat')
    wave = annots['epoch_data']
    wave = np.array(wave[:-1])
    x.append(wave.T)

x = np.array(x)

print('X shape=',x.shape, 'Y shape=', y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=43, stratify=y)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

print('Model Training')

model = keras.models.Sequential()
model.add(keras.layers.LSTM(128, activation='tanh', return_sequences=True, input_shape=(1000, 64)))
model.add(keras.layers.LSTM(512, activation='tanh', return_sequences = True))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.LSTM(256, activation='tanh', return_sequences=True))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.LSTM(128, activation='tanh'))

model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(20, activation='softmax'))

print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=25, validation_data=(x_test, y_test), batch_size=34)
model.save("LSTM_model.h5")

print('Plotting Model Graph')

pd.DataFrame(history.history).plot(figsize = (8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.savefig('LSTM_graph.png')
plt.show()

print('The End')