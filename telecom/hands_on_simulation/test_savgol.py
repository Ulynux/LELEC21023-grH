import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt, sawtooth

""""
# Génération du signal d'exemple (bruité)
np.random.seed(42)  # Pour la reproductibilité
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.randn(100)  # Sinusoïde avec bruit

# Paramètres des fenêtres
window_sizes = range(5, 25, 5)
poly_order = 3  # Ordre du polynôme

# Création du subplot (2 lignes, 2 colonnes)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Boucle sur les différentes tailles de fenêtre
for i, ax in zip(window_sizes, axes.ravel()):
    y_smooth = savgol_filter(y, i, poly_order)
    
    # Tracé des courbes
    ax.plot(x, y, label="Signal bruité", alpha=0.5)
    ax.plot(x, y_smooth, label=f"Signal lissé (Fenêtre={i})", linewidth=2)
    
    # Ajout du titre et de la légende
    ax.set_title(f"Fenêtre = {i}")
    ax.legend()

# Ajustement de l'affichage
plt.tight_layout()
plt.show()

window_size2 = 15  # Taille de la fenêtre (fixe)
poly_order_2 = range(1, 5)  # Différents ordres du polynôme

# Création du subplot (2 lignes, 2 colonnes)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Boucle sur les différents ordres de polynôme
for i, ax in zip(poly_order_2, axes.ravel()):
    y_smooth = savgol_filter(y, window_size2, i)  # Correction ici

    #derive the smoothed signal
    y_smooth2 = np.gradient(y_smooth, x)
    
    # Tracé des courbes
    ax.plot(x, y, label="Signal bruité", alpha=0.5)
    ax.plot(x, y_smooth, label=f"Signal lissé (Ordre {i})", linewidth=2)
    
    # Ajout du titre et de la légende
    ax.set_title(f"Ordre du polynôme = {i}")
    ax.legend()

# Ajustement de l'affichage
plt.tight_layout()
plt.show()
"""

x = np.linspace(0, 10, 100)
y = sawtooth(2 * np.pi * x, width=0.5) + 0.5 * np.random.randn(100)
# Paramètres pour le lissage
w = 20  # Taille de la fenêtre
p = 1  # Ordre du polynôme

# Lissage du signal
y_smooth = savgol_filter(y, w, p)

# Première dérivée du signal lissé
y_first_derivative = np.gradient(y_smooth, x)

# Deuxième dérivée du signal lissé
y_second_derivative = np.gradient(y_first_derivative, x)

# Création du subplot (2 lignes, 2 colonnes)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Tracé du signal original
axes[0, 0].plot(x, y, label="Signal bruité", alpha=0.5)
axes[0, 0].set_title("Signal bruité")
axes[0, 0].legend()

# Tracé du signal lissé
axes[0, 1].plot(x, y_smooth, label="Signal lissé", linewidth=2)
axes[0, 1].set_title("Signal lissé")
axes[0, 1].legend()

# Tracé de la première dérivée
axes[1, 0].plot(x, y_first_derivative, label="Première dérivée", linewidth=2)
axes[1, 0].set_title("Première dérivée")
axes[1, 0].legend()

# Tracé de la deuxième dérivée
axes[1, 1].plot(x, y_second_derivative, label="Deuxième dérivée", linewidth=2)
axes[1, 1].set_title("Deuxième dérivée")
axes[1, 1].legend()

# Ajustement de l'affichage
plt.tight_layout()
plt.show()


#compute the second derivative of the smoothed signal 


###############
# Butterworth # 
###############

"""
x = np.linspace(0, 10, 100)
y = np.sin(x) + 0.5 * np.random.randn(100)  # Sinusoïde avec bruit

# 2. Fonction de filtrage Butterworth
def butterworth_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs  # Fréquence de Nyquist
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)  # Filtrage avant-arrière pour éviter le décalage de phase

fs = 10   # Fréquence d'échantillonnage
cutoff = 1  # Fréquence de coupure
y_butter = butterworth_filter(y, cutoff, fs)

# 3. Affichage du signal original vs filtré
plt.figure(figsize=(10, 5))
plt.plot(x, y, label="Signal bruité", alpha=0.5)
plt.plot(x, y_butter, label="Signal filtré Butterworth", linewidth=2)
plt.title("Filtrage Butterworth")
plt.legend()
plt.show()
"""



