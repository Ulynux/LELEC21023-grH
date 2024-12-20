import pandas as pd
import matplotlib.pyplot as plt

# Lecture du fichier CSV en ignorant les lignes d'en-tête inutiles
# Remplacez par le chemin réel
file_path = "power_measurement\R6\DATA.csv"

df = pd.read_csv(file_path, skiprows=7,encoding = 'latin1')
df.columns = df.columns.str.strip()
# Extraction des colonnes utiles
time = df['Time(S)']
ch1 = df['CH1(V)']
ch2 = df['CH2(V)']
# Calcul de puissance
R = 20
Vin = 3.3
puissance = ch1/R * Vin
puissance_mW = puissance*1000

# Debug
"""
print('------------')
print(ch1)
print(puissance)
"""

# Création des graphiques
""""
plt.figure(figsize=(10, 6))
plt.plot(time, ch1, color='b')

plt.title('Power consumption')
plt.xlabel('Temps (s)')

plt.ylabel('Tension(V)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
"""
# Création du graphique
plt.figure(figsize=(10, 6))
plt.plot(time, puissance_mW, color='r')

plt.title('Power consumption')
plt.xlabel('Time (s)')

plt.ylabel('Power (mW)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0.8,2.6)

# Lignes en tiret
plt.axvline(x=0.965, color = 'k', label = 'axvline - full height',linestyle='dashed')
plt.axvline(x=1.964, color = 'k', label = 'axvline - full height',linestyle='dashed')
plt.axvline(x=2.266, color = 'k', label = 'axvline - full height',linestyle='dashed')
plt.axvline(x=2.364, color = 'k', label = 'axvline - full height',linestyle='dashed')

# Une fois pret
#plt.savefig("power_measurement\Power_Usage_MCU.png")
plt.show()
print('FIN')