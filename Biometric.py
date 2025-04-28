print('start')
import pandas as pd
import numpy as np
import scipy
from scipy.io import loadmat
import matplotlib.pyplot as plt
from librosa.feature import mfcc, melspectrogram
from librosa.display import specshow
import librosa

print("Reading CSV file")
df = pd.read_csv("Enrollment_Info.csv")

print("Subject Dictionary")
subs = {'sub011':[]}


print("Epoch Subject Mapping")
for epochid, subid in zip(df['EpochID'], df['subject']):
    if subid == 'sub011':
        subs[subid].append(epochid)


print("Mel spectrogram creation")
for subid in subs.keys():
    for epochid in subs[subid]:
        print(epochid, end='\n')
        annots = loadmat('Enrollment/'+epochid+'.mat')
        wave = annots['epoch_data']

        wave = wave.flatten()

        mel_spec = librosa.feature.melspectrogram(y=wave, n_mels=128)

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        plt.figure(figsize=(14, 8))
        specshow(mel_spec_db, y_axis='log', fmax=8000, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()
        plt.savefig('Mel Spectrograms/'+subid+'/'+epochid+'.png')
        plt.close()
print("The End")