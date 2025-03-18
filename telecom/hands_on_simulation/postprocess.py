import matplotlib.pyplot as plt
import numpy as np
from chain import Chain
from scipy.signal import firwin, freqz
from scipy.special import erfc
import PER as per   


plots_folder = "telecom/plots/"
test_name = ["h_old", "h1", "h2", "h3", "h4"]


# Variables needed

# Chain
chain = Chain()
R = chain.osr_rx
B = chain.bit_rate
fs = B * R
# Lowpass filter taps
taps = firwin(chain.numtaps, 130000, fs=fs)

# Theoretical curves - normalization
Cu = np.correlate(taps, taps, mode="full")  # such that Cu[len(taps)-1] = 1
sum_Cu = 0
for r in range(0, R):
    for rt in range(0, R):
        sum_Cu += Cu[len(taps) - 1 + r - rt]
shift_SNR_out = 10 * np.log10(R**2 / sum_Cu)  # 10*np.log10(chain.osr_rx)
shift_SNR_filter = 10 * np.log10(1 / np.sum(np.abs(taps) ** 2))


# Read files:
SNRs_dB = []
SNRs_dB_shifted = []
BER = []
PER = []
RMSE_cfo = []
RMSE_sto = []
preamble_mis = []
preamble_false = []

for name in test_name:
    data = np.loadtxt(f"telecom/plots/{name}.csv", delimiter="\t")
    SNRs_dB.append(data[:, 0])
    SNRs_dB_shifted.append(data[:, 1])
    BER.append(data[:, 2])
    PER.append(data[:, 3])
    RMSE_cfo.append(data[:, 4])
    RMSE_sto.append(data[:, 5])
    preamble_mis.append(data[:, 6])
    preamble_false.append(data[:, 7])


savefig = False


def plots():
    fig, ax = plt.subplots(constrained_layout=True)
    for i in range(len(test_name)):
        ax.plot(SNRs_dB[i], BER[i], "-s", label=test_name[i])
    ax.set_ylabel("BER")
    ax.set_xlabel("SNR$_{o}$ [dB]")
    ax.set_yscale("log")
    ax.set_ylim((1e-6, 1))
    ax.set_xlim((0, 30))
    ax.grid(True)
    ax.set_title("Average Bit Error Rate")
    ax.legend()
    if savefig: plt.savefig(plots_folder+"BER_from_file.png")

    # PER
    # fig, ax = plt.subplots(constrained_layout=True)
    # ax.plot(per.SNR_aver - shift_SNR_filter,per.PACKET_ERROR,label="Measurements")
    # ax.plot(SNRs_dB_shifted, PER, "-s", label="Simulation")
    # ax.set_ylabel("PER")
    # ax.set_xlabel("SNR$_{o}$ [dB]")
    # ax.set_yscale("log")
    # ax.set_ylim((1e-2, 1))
    # ax.set_xlim((0, 20))
    # ax.grid(True)
    # ax.set_title("Average Packet Error Rate")
    # ax.legend()
    # if savefig: plt.savefig(plots_folder+"PER_from_file.png")

    # Preamble detection
    plt.figure()
    plt.plot(SNRs_dB, preamble_mis * 100, "-s", label="Miss-detection")
    plt.plot(SNRs_dB, preamble_false * 100, "-s", label="False-detection")
    plt.title("Preamble detection error ")
    plt.ylabel("[%]")
    plt.xlabel("SNR [dB]")
    plt.ylim([-1, 101])
    plt.grid()
    plt.legend()
    if savefig: plt.savefig(plots_folder+"Preamble_detection_from_file.png")

    # CFO 
    plt.figure()
    plt.semilogy(SNRs_dB, RMSE_cfo, "-s")
    plt.title("RMSE CFO")
    plt.ylabel("RMSE [-]")
    plt.xlabel("SNR [dB]")
    plt.grid()
    if savefig: plt.savefig(plots_folder+"RMSE_CFO_from_file.png")

    # STO
    plt.figure()
    plt.semilogy(SNRs_dB, RMSE_sto, "-s")
    plt.title("RMSE STO")
    plt.ylabel("RMSE [-]")
    plt.xlabel("SNR [dB]")
    plt.grid()
    if savefig: plt.savefig(plots_folder+"RMSE_STO_from_file.png")


def modulation_index():

    fig, ax = plt.subplots(constrained_layout=True)
    for i in range(len(test_name)):
        ax.plot(SNRs_dB_shifted[i], BER[i], "-s", label=test_name[i])
    ax.set_xlabel("SNR$_{o}$ [dB]")
    ax.set_yscale("log")
    ax.set_ylim((1e-6, 1))
    ax.set_xlim((0, 30))
    ax.grid(True)
    ax.set_title("Average Bit Error Rate")
    ax.legend()
    plt.savefig("telecom/plots/BER_comparison.png")

    plt.figure()
    for i in range(len(test_name)):
        plt.plot(SNRs_dB[i], preamble_mis[i] * 100, "-s", label=f"{test_name[i]} Miss-detection")
        plt.plot(SNRs_dB[i], preamble_false[i] * 100, "-s", label=f"{test_name[i]} False-detection")
    plt.title("Preamble detection error ")
    plt.ylabel("[%]")
    plt.xlabel("SNR [dB]")
    plt.ylim([-1, 101])
    plt.grid()
    plt.legend()
    plt.savefig("telecom/plots/Preamble_detection_comparison.png")

    plt.figure()
    for i in range(len(test_name)):
        plt.semilogy(SNRs_dB[i], RMSE_cfo[i] * 100, "-s", label=f"{test_name[i]} RMSE CFO")
    plt.title("RMSE CFO")
    plt.ylabel("RMSE [-]")
    plt.xlabel("SNR [dB]")
    plt.grid()
    plt.legend()
    plt.savefig("telecom/plots/CFO_comparison.png")

    plt.figure()
    for i in range(len(test_name)):
        plt.semilogy(SNRs_dB[i], RMSE_sto[i], "-s", label=f"{test_name[i]} RMSE STO")
    plt.title("RMSE STO")
    plt.ylabel("RMSE [-]")
    plt.xlabel("SNR [dB]")
    plt.grid()
    plt.legend()
    plt.savefig("telecom/plots/STO_comparison.png")

import matplotlib.pyplot as plt
import numpy as np
SNR = [-2.78502413, -1.78502413, -0.78502413, 0.21497587, 1.21497587, 2.21497587, 
       3.21497587, 4.21497587, 5.21497587, 6.21497587, 7.21497587, 8.21497587, 
       9.21497587, 10.21497587, 11.21497587, 12.21497587, 13.21497587, 14.21497587, 
       15.21497587, 16.21497587, 17.21497587, 18.21497587, 19.21497587, 20.21497587, 
       21.21497587, 22.21497587, 23.21497587, 24.21497587, 25.21497587, 26.21497587, 
       27.21497587, 28.21497587, 29.21497587, 30.21497587, 31.21497587]

BER_130000 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.27625, 0.0675, 0.03, 
              0.02625, 0.025, 0.0175, 0.01625, 0., 0., 0., 0., 0., 
              0., 0., 0., 0., 0., 0., 0., 0., 0., 
              0., 0., 0., 0., 0., 0., 0., 0.]

BER_120000 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.36125, 0.12625, 0.0825, 0.04125, 
              0.03125, 0.00625, 0.0025, 0., 0., 0., 0., 0., 0., 
              0., 0., 0., 0., 0., 0., 0., 0., 0., 
              0., 0., 0., 0., 0., 0., 0., 0.]

BER_110000 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.345, 0.13, 0.12125, 0.0825, 
              0.04625, 0.08, 0.02625, 0., 0., 0., 0., 0., 0., 
              0., 0., 0., 0., 0., 0., 0., 0., 0., 
              0., 0., 0., 0., 0., 0., 0., 0.]

BER_100000 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.15125, 0.1025, 0.06375, 0.02875, 
              0.0025, 0.00125, 0.00125, 0., 0., 0., 0., 0., 0., 
              0., 0., 0., 0., 0., 0., 0., 0., 0., 
              0., 0., 0., 0., 0., 0., 0., 0.]

BER_90000 = [0.5, 0.5, 0.5, 0.5, 0.5, 0.25875, 0.1575, 0.10625, 0.05375, 
             0.10125, 0.02375, 0., 0., 0., 0., 0., 0., 0., 
             0., 0., 0., 0., 0., 0., 0., 0., 0., 
             0., 0., 0., 0., 0., 0., 0., 0.]

BER_80000 = [0.5, 0.5, 0.5, 0.5, 0.395, 0.32625, 0.22375, 0.12, 0.11125, 
             0.005, 0.00125, 0., 0., 0., 0., 0., 0., 0., 
             0., 0., 0., 0., 0., 0.05, 0.05, 0.1, 0.15, 
             0.15, 0.15, 0.1, 0.15, 0.15, 0.15, 0.15, 0.15]

BER_70000 = [0.5, 0.5, 0.5, 0.4375, 0.32625, 0.18125, 0.08625, 0.075, 0.05625, 
             0.05125, 0.02625, 0.07625, 0.0725, 0.0675, 0.21, 0.26, 0.25875, 0.3, 
             0.3, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 
             0.35, 0.35, 0.35, 0.35, 0.35, 0.4, 0.4, 0.4]

BER_60000 = [0.5, 0.5, 0.4625, 0.39375, 0.2275, 0.18, 0.1625, 0.1175, 0.18125, 
             0.2225, 0.20375, 0.2275, 0.21, 0.32125, 0.35125, 0.4, 0.5, 0.5, 
             0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 
             0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

BER_50000 = [0.5, 0.5, 0.48375, 0.4275, 0.3975, 0.415, 0.46, 0.48375, 0.5, 
             0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 
             0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 
             0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
# Tracer les courbes BER
plt.figure(figsize=(10, 6))

plt.plot(SNR, BER_130000, label="BER 130k", marker='o')
plt.plot(SNR, BER_120000, label="BER 120k", marker='s')
plt.plot(SNR, BER_110000, label="BER 110k", marker='^')
plt.plot(SNR, BER_100000, label="BER 100k", marker='x')
plt.plot(SNR, BER_90000, label="BER 90k", marker='D')


plt.yscale('log')
plt.xlabel('SNR (dB)')
plt.ylabel('BER (Log Scale)')
plt.xlim(-3,15)
plt.title('BER en fonction de SNR pour différents taux')
plt.legend()
plt.grid(True)
plt.savefig("telecom/plots/BER_differents_cutoff.png")
plt.show()

