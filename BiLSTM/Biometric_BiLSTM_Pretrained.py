print('start')

import pandas as pd
import numpy as np
import scipy
from scipy.io import loadmat
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

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
    annots = loadmat('Enrollment/'+epochid+'.mat')
    wave = annots['epoch_data']
    wave = np.array(wave[:-1])
    x.append(wave.T)

x = np.array(x)

print('X shape=',x.shape, 'Y shape=', y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=43, stratify=y)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

print('Model Training')

model = keras.models.load_model('BiLSTM_Model.h5')

print(model.summary())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('BiLSTM_Model.h5', save_best_only=True)

history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), callbacks=[checkpoint_cb])

print('Plotting Model Graph')

y_pred = model.predict(x_test)
pred = []
for i in y_pred:
    pred.append(np.argmax(i))

test = []
for i in y_test:
    test.append(np.argmax(i))

df_hist = pd.DataFrame(history.history)
df_hist.to_csv("BiLSTM_History3.csv")
df_hist.plot(figsize = (8,5))

plt.grid(True)
plt.gca().set_ylim(0,1)
plt.savefig('BiLSTM_graph.png')
plt.show()

print('The End')