import matplotlib.pyplot as plt
import numpy as np
from chain import Chain
from scipy.signal import firwin, freqz
from scipy.special import erfc
import PER as per   


plots_folder = "telecom/plots/"
test_name = ["M8", "M16", "M32", "M64", "M128"]

R = [8, 16, 32, 64, 128]

chain = Chain()

SNRs_dB = chain.snr_range
R = chain.osr_rx
print("R = ",R)
B = chain.bit_rate
fs = B * R

taps = firwin(chain.numtaps, 130000, fs=fs)
fcutoff = [90000, 100000, 110000, 120000, 130000]

# Read files:
SNRs_dB = []
SNRs_dB_shifted = []
BER = []
PER = []
RMSE_cfo = []
RMSE_sto = []
preamble_mis = []
preamble_false = []


savefig = True


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


#ecris une fonction plus les data de 90000Hz, 100000Hz, 110000Hz, 120000Hz, 130000Hz

def plot_cutoff_frequency():

    # Lowpass filter taps
    

    fig, ax = plt.subplots(constrained_layout=True)

    for i in range(len(test_name)):

        filename = f"/home/matthieu/MASTER1/Q2/LELEC21023-grH/telecom/plots/{test_name[i]}.csv"

        data = np.genfromtxt(filename, delimiter=",", skip_header=1)   

        taps = firwin(chain.numtaps, fcutoff[i], fs=fs)

        # Theoretical curves - normalization
        Cu = np.correlate(taps, taps, mode="full")
        sum_Cu = 0
        for r in range(0, R):
            for rt in range(0, R):
                sum_Cu += Cu[len(taps) - 1 + r - rt]
        shift_SNR_out = 10 * np.log10(R**2 / sum_Cu)
        shift_SNR_filter = 10 * np.log10(1 / np.sum(np.abs(taps) ** 2))

        SNR = data[:, 0]
        PER = data[:, 1]

        plt.plot(SNR -shift_SNR_filter + shift_SNR_out, PER, "-s", label=f"{test_name[i]} Hz")

    plt.xlabel("SNR$_{o}$ [dB]")
    plt.yscale("log")
    plt.ylim((10e-3, 1))
    plt.xlim((5, 30))
    plt.grid(True)
    plt.title("Average Bit Error Rate")
    plt.legend()
    plt.savefig("plots/BER_cutoff_frequency.png")
    plt.show()

def plot_oversampling_factor():

    PER_8 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.2, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    print(len(PER_8))


    SNR = [-3.08066884, -2.08066884, -1.08066884, -0.08066884, 0.91933116, 1.91933116,
        2.91933116, 3.91933116, 4.91933116, 5.91933116, 6.91933116, 7.91933116,
        8.91933116, 9.91933116, 10.91933116, 11.91933116, 12.91933116, 13.91933116,
        14.91933116, 15.91933116, 16.91933116, 17.91933116, 18.91933116, 19.91933116,
        20.91933116, 21.91933116, 22.91933116, 23.91933116, 24.91933116, 25.91933116,
        26.91933116, 27.91933116, 28.91933116, 29.91933116, 30.91933116]

    print(len(SNR))


    PER_16 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
        0.6, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    PER_32 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
       0.0, 0.0, 0.0, 0.0, 0.0]

    PER_64 = [1.0, 1.0, 1.0, 1.0, 1.0, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
       0.0, 0.0, 0.0, 0.0, 0.0]


    #plot les resultats

    plt.figure()
    plt.plot(SNR, PER_8, "-s", label="8")
    plt.plot(SNR, PER_16, "-s", label="16")
    plt.plot(SNR, PER_32, "-s", label="32")
    plt.plot(SNR, PER_64, "-s", label="64")
    plt.title("PER")
    plt.ylabel("PER")
    plt.xlabel("SNR [dB]")
    plt.grid()
    plt.legend()
    plt.yscale("log")
    plt.savefig("plots/PER_oversampling_factor.png")
    plt.show()
 




plot_oversampling_factor()


