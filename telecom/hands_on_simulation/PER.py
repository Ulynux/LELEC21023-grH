from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def extract_data(file_path):
    try:
        with open(file_path, 'r') as file:
            data = file.readlines()
            extracted_data = []
            ignored_lines = 0
            
            for line in data[1:]:
                parts = line.strip().split()
                cfo = float(parts[0])
                snr = float(parts[2])
                txp = int(parts[3])
                ber = float(parts[4])
                invalid = int(parts[5])
                extracted_data.append((cfo, snr, txp, ber, invalid))

            

            
        return extracted_data
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")



file_path = 'Data.txt'

# Remplace par le chemin réel de ton fichier
data_exp = extract_data(file_path)
print(len(data_exp))
cfo = []
snr = []
txp = []
ber = []
invalid = []


cfo, snr, txp, ber, invalid = zip(*data_exp)


txp_count = dict(Counter(txp))

key = list(txp_count.keys())
occurence = []
for i in range(len(key)):
    occurence.append(txp_count.get(key[i]))



SNR = []
INVALID = []
CFO = []
index = 0



snr = list(snr)
invalid = list(invalid)
cfo = list(cfo)

for i in range(0,len(occurence)):
    SNR.append(snr[:occurence[i]])
    del snr[:occurence[i]]
    INVALID.append(invalid[:occurence[i]])
    del invalid[:occurence[i]]

    

    packet = 0
for i in range(len(INVALID)):
    if(len(INVALID[i]) != 1000):
        for j in range(1000 - len(INVALID[i])):
            INVALID[i].append(1)

SNR_aver = []
PACKET_ERROR = []
for i in range(len(SNR)):
    sum = 0
    for j in range(len(SNR[i])):
        sum += SNR[i][j]
    SNR_aver.append(sum/len(SNR[i]))

for i in range(len(INVALID)):
    packet = 0
    for j in range(len(INVALID[i])):
        packet+=INVALID[i][j]
    PACKET_ERROR.append(packet/1000)

PACKET_ERROR[1] -= 0.005 # J'ai appuyé sur le bouton detect trop tard


PER = []
PER.append(np.mean(PACKET_ERROR[:4]))
PER.append(np.mean(PACKET_ERROR[4:6]))
PER += PACKET_ERROR[6:]
PER.append(0)
PER = np.array(PER)

SNR = []
SNR.append(np.mean(SNR_aver[:4]))
SNR.append(np.mean(SNR_aver[4:6]))
SNR += SNR_aver[6:]
SNR.append(20)
SNR = np.array(SNR)

print("SNR : ",SNR_aver)
print("PER : ",PACKET_ERROR)
print("PER : ",PER)
print("SNR : ",SNR)



plt.figure()
plt.plot(SNR, PER, marker='o', linestyle = '-')
plt.grid(True)
plt.title('PER vs SNR')
plt.xlabel('SNRe [dB]')
plt.ylabel('PER [-]')
plt.yscale('log')
plt.xlim(0, 20)
plt.ylim(1e-3, 1)
plt.savefig('telecom/PER_global_chain.pdf')
plt.show()


sorted_indices = np.argsort(np.array(SNR_aver))
SNR_aver = np.array(SNR_aver)[sorted_indices]
PACKET_ERROR = np.array(PACKET_ERROR)[sorted_indices]
plt.figure()
plt.plot(SNR_aver, PACKET_ERROR, marker='o', linestyle = '-')
plt.grid(True)
plt.title('PER vs SNR')
plt.xlabel('SNRe [dB]')
plt.ylabel('PER')
plt.yscale('log')
plt.ylim(1e-3, 1e-1)
plt.xlim(9, 17)
plt.savefig('telecom/PER_zoom.pdf')
plt.show()

