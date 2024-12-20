import pandas as pd
import matplotlib.pyplot as plt

# Lecture du fichier CSV en ignorant les lignes d'en-tête inutiles
file_path = '/home/matthieu/Téléchargements/RADIO.csv'  # Remplacez par le chemin réel

df = pd.read_csv(file_path, skiprows=7,encoding = 'latin1')
df.columns = df.columns.str.strip()
# Extraction des colonnes utiles
time = df['Time(S)']
ch1 = df['CH1(V)']
ch2 = df['CH2(V)']

# Création du graphique
plt.figure(figsize=(10, 6))
plt.plot(time, ch1, label='CH1(V)', color='b')


plt.title('Power consumption')
plt.xlabel('Temps (s)')

plt.ylabel('Tension (V)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.savefig('/home/matthieu/MASTER1/LELEC21023-grH/plots/Radio.png')
plt.show()