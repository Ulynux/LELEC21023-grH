import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq

import matplotlib.pyplot as plt

# Lire les données du fichier CSV
file_path = '/home/louis/Documents/EmbeddedProject/LELEC21023-grH/mcu/hands_on_audio_acquisition/csv_files/acq-1.csv'
data = pd.read_csv(file_path)

# Extraire les colonnes de temps et de tension
time = data['Time (s)']
voltage = data['Voltage (mV)']

# Calculer la FFT
N = len(time)
T = time[1] - time[0]  # Intervalle de temps entre les échantillons
yf = fft(voltage.to_numpy())
xf = fftfreq(N, T)[:N//2]

# Tracer le signal original
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time, voltage)
plt.title('Signal Original')
plt.xlabel('Temps (s)')
plt.ylabel('Tension (mV)')

# Tracer la FFT
plt.subplot(2, 1, 2)
plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.title('FFT du Signal')
plt.xlabel('Fréquence (Hz)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()