from collections import Counter
import matplotlib.pyplot as plt


def extract_data(file_path):
    # Lire le fichier TXT et extraire les colonnes spécifiques
    try:
        with open(file_path, 'r') as file:
            data = file.readlines()
            extracted_data = []
            for line in data:
                parts = line.strip().split()
                if len(parts) >= 6 and parts[0].replace('.', '', 1).isdigit():
                    cfo = float(parts[0])
                    snr = float(parts[2])
                    txp = int(parts[3])
                    ber = float(parts[4])
                    invalid = int(parts[5])
                    extracted_data.append((cfo, snr, txp, ber, invalid))
        return extracted_data
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")


file_path = 'telecom/Data.txt'
# Remplace par le chemin réel de ton fichier
data_exp = extract_data(file_path)

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
print(occurence)
for i in range(0,len(occurence)):
    SNR.append(snr[:occurence[i]])
    del snr[:occurence[i]]
    INVALID.append(invalid[:occurence[i]])
    del invalid[:occurence[i]]

    

    packet = 0
for i in range(len(INVALID)):
    if(len(INVALID[i]) != 100):
        for j in range(100 - len(INVALID[i])):
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
    PACKET_ERROR.append(packet/100)


# plt.figure()
# plt.hist(cfo,bins = 20)
# plt.xlabel("CFO")
# plt.ylabel("OCCURENCe")
# plt.savefig('telecom/CFO.png')


# plt.figure()
# plt.plot(SNR_aver,PACKET_ERROR,'o',linestyle = '-')
# plt.yscale('log')
# plt.grid(True)
# plt.xlabel('SNR')
# plt.ylabel('PER')
# plt.savefig('telecom/test_packet.png')

# plt.show()

