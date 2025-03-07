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




modulation_index()