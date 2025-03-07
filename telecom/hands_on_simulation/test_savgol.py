import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_coeffs

# Paramètres du filtre
window_length = 51 # Doit être impair
polyorder = 3 # Ordre du polynôme

# Obtenir les coefficients du filtre
coeffs = savgol_coeffs(window_length, polyorder)

# Affichage de la fenêtre du filtre
plt.stem(np.arange(-(window_length // 2), (window_length // 2) + 1), coeffs, basefmt=" ")
plt.xlabel("Index")
plt.ylabel("Amplitude")
plt.title(f"Fenêtre du filtre de Savitzky-Golay\n(Window Length={window_length}, Polyorder={polyorder})")
plt.grid()
plt.show()




