import pandas as pd
import matplotlib.pyplot as plt

# Lecture du fichier CSV en ignorant les lignes d'en-tête inutiles
file_path = 'power_measurement/DATA_POWER/Total_power_ok.csv'  # Remplacez par le chemin réel

df = pd.read_csv(file_path, skiprows=7,encoding = 'latin1')
df.columns = df.columns.str.strip()

# Extraction des colonnes utiles

time = df['Time(S)'][3500:60000]
ch1 = df['CH1(V)'][3500:60000] + 0.026
power = ch1/10 * 3.3 * 1000    # in mW, I*U, I = ch1/R, R = 10 Ohm, U = 3.3V

# Création du graphique
plt.figure(figsize=(10, 6))
plt.plot(time, power)
plt.title('Total power consumption')
plt.xlabel('Temps (s)')
plt.ylabel('Power (mW)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

time = df['Time(S)'][1800:17000]
ch1 = df['CH1(V)'][1800:17000] + 0.026
power = ch1/10 * 3.3 * 1000    # in mW, I*U, I = ch1/R, R = 10 Ohm, U = 3.3V

# Création du graphique
plt.figure(figsize=(10, 6))
plt.plot(time, power)
plt.title('Total power consumption')
plt.xlabel('Temps (s)')
plt.ylabel('Power (mW)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.axvline(x=-0.5, color='r', linestyle='--')
plt.axvline(x=0.501, color='r', linestyle='--')
plt.axvline(x=0.596, color='r', linestyle='--')
plt.axvline(x=0.73, color='r', linestyle='--')
plt.text(0.0, 35, '1', color='r')
plt.text(0.543, 20, '2', color='r')
plt.text(0.654, 20, '3', color='r')
plt.show()



# Lecture du fichier CSV en ignorant les lignes d'en-tête inutiles
file_path = 'power_measurement/DATA_POWER/MCU_power_ok.csv'  # Remplacez par le chemin réel

df = pd.read_csv(file_path, skiprows=7, encoding = 'latin1')
df.columns = df.columns.str.strip()

# Extraction des colonnes utiles

time = df['Time(S)'][:16000] + 0.301
ch1 = df['CH1(V)'][:16000] + 0.026
power = ch1/10 * 3.3 * 1000    # in mW, I*U, I = ch1/R, R = 10 Ohm, U = 3.3V

# Création du graphique
plt.figure(figsize=(10, 6))
plt.plot(time, power)
plt.title('MCU power consumption')
plt.xlabel('Temps (s)')
plt.ylabel('Power (mW)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.axvline(x=-0.5, color='r', linestyle='--')
plt.axvline(x=0.501, color='r', linestyle='--')
plt.axvline(x=0.596, color='r', linestyle='--')
plt.axvline(x=0.73, color='r', linestyle='--')
plt.show()


# Lecture du fichier CSV en ignorant les lignes d'en-tête inutiles
file_path = 'power_measurement/DATA_POWER/DATA_AFE_1200.csv'  # Remplacez par le chemin réel

df = pd.read_csv(file_path, skiprows=7, encoding = 'latin1')
df.columns = df.columns.str.strip()

# Extraction des colonnes utiles

time = df['Time(S)']
ch1 = df['CH1(V)'] + 0.026
current = ch1/1.2e3 *1e6    # in mW, I*U, I = ch1/R, R = 10 Ohm, U = 3.3V 

# Création du graphique
plt.figure(figsize=(10, 6))
plt.plot(time, current)
plt.title('AFE current consumption')
plt.xlabel('Temps (s)')
plt.ylabel('Current (uA)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# plt.figure(figsize=(10, 6))
# plt.plot(time, power)
# plt.title('AFE power consumption')
# plt.xlabel('Temps (s)')
# plt.ylabel('Current (uA)')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()



# Lecture du fichier CSV en ignorant les lignes d'en-tête inutiles
file_path = 'power_measurement/DATA_POWER/Radio_power_ok.csv'  # Remplacez par le chemin réel

df = pd.read_csv(file_path, skiprows=7, encoding = 'latin1')
df.columns = df.columns.str.strip()

# Extraction des colonnes utiles

time = df['Time(S)'][4500:61000]
ch1 = df['CH1(V)'][4500:61000] + 0.026
power = ch1/10 * 3.3 * 1000    # in mW, I*U, I = ch1/R, R = 10 Ohm, U = 3.3V

# Création du graphique
plt.figure(figsize=(10, 6))
plt.plot(time, power)
plt.title('Radio power consumption')
plt.xlabel('Temps (s)')
plt.ylabel('Power (mW)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print('FIN')