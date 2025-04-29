import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import librosa

# Load the audio signal
signal, sr = librosa.load('classification/src/classification/datasets/tmp_soundfiles/chainsaw_01.wav', sr=None)

# Generate the time array based on the signal length and sampling rate
time = np.linspace(0, len(signal) / sr, len(signal))  # Time array matching the signal length

# Split the signal into two halves
midpoint = len(signal) // 2
first_half = signal[:midpoint]
second_half = signal[midpoint:]

# Create a grid layout for the plots
fig = plt.figure(figsize=(9, 6))
gs = GridSpec(2, 2, height_ratios=[1.5, 1])  # Adjust height ratios to make the first plot less large

# Plot the Chainsaw signal (spanning the top row)
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(time, signal, label="Chainsaw Signal")
ax1.axvline(x=time[midpoint], color="red", linestyle="--", label="Midpoint")  # Add vertical line at midpoint
ax1.set_title("Chainsaw Signal")
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Amplitude [V]")
ax1.legend()

# Plot the first half of the signal (bottom-left)
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(time[:midpoint], first_half, label="First Half", color="orange")
ax2.set_title("Recorded First Half of the Signal")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Amplitude [V]")
ax2.legend()

# Plot the second half of the signal (bottom-right)
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(time[midpoint:], second_half, label="Second Half", color="green")
ax3.set_title("Recorded Second Half of the Signal")
ax3.set_xlabel("Time [s]")
ax3.set_ylabel("Amplitude [V]")
ax3.legend()

# Save the plot as a PDF file
plt.tight_layout()
plt.savefig("plotsignal.pdf", format="pdf")
plt.show()