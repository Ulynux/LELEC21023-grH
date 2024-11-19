import numpy as np
import matplotlib.pyplot as plt

def plot_combined_from_csv(file_paths):
    plt.figure()
    
    for file_path in file_paths:
        # Read data from CSV file
        data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
        SNRs_dB = data[:, 0]
        RMSE_cfo = data[:, 1]

        # Extract N value from the file name
        N_value = file_path.split('N=')[1].split('.')[0]

        # Plot data
        plt.semilogy(SNRs_dB, RMSE_cfo, "-s", label=f"N={N_value}")

    plt.title("RMSE CFO for Different N Values")
    plt.ylabel("RMSE [-]")
    plt.xlabel("SNR [dB]")
    plt.grid()
    plt.legend()
    plt.savefig("plots/RMSE_CFO_combined.png")

# Example usage
if __name__ == "__main__":
    file_paths = [
        'plots/RMSE_CFO_dataN=1.csv',
        'plots/RMSE_CFO_dataN=2.csv',
        'plots/RMSE_CFO_dataN=4.csv',
        'plots/RMSE_CFO_dataN=8.csv',
        'plots/RMSE_CFO_dataN=16.csv',
        'plots/RMSE_CFO_dataN=32.csv'
    ]
    plot_combined_from_csv(file_paths)