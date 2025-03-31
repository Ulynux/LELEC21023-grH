#include <adc_dblbuf.h>
#include "config.h"
#include "main.h"
#include "spectrogram.h"
#include "arm_math.h"
#include "utils.h"
#include "s2lp.h"
#include "packet.h"


static volatile uint16_t ADCDoubleBuf[2*ADC_BUF_SIZE]; /* ADC group regular conversion data (array of data) */
static volatile uint16_t* ADCData[2] = {&ADCDoubleBuf[0], &ADCDoubleBuf[ADC_BUF_SIZE]};
static volatile uint8_t ADCDataRdy[2] = {0, 0};

static volatile uint8_t cur_melvec = 0;
static q15_t mel_vectors[N_MELVECS][MELVEC_LENGTH];
static q15_t power_threshold = 100;
static uint32_t packet_cnt = 0;

static volatile int32_t rem_n_bufs = 0;

int StartADCAcq(int32_t n_bufs) {
	// argument n_bufs = N_MELVECS = 20 = le nombre de buffers à acquérir
	// 
	rem_n_bufs = n_bufs;
	cur_melvec = 0;
	if (rem_n_bufs != 0) {
		return HAL_ADC_Start_DMA(&hadc1, (uint32_t *)ADCDoubleBuf, 2*ADC_BUF_SIZE);
	} else {
		return HAL_OK;
	}
}

uint32_t get_signal_power(uint16_t *buffer, size_t len){
	uint64_t sum = 0;
	uint64_t sum2 = 0;
	for (size_t i=0; i<len; i++) {
		sum += (uint64_t) buffer[i];
		sum2 += (uint64_t) buffer[i]*(uint64_t) buffer[i];
	}
	return (uint32_t)(sum2/len - sum*sum/len/len);
}
static int power_threshold_reached() {
	uint32_t power = get_signal_power((uint16_t *)ADCData[0], ADC_BUF_SIZE);
	if (power > power_threshold) {
		power_threshold = power;
		DEBUG_PRINT("Enough power in the sound to compute it :) \r\n");

		return 1;
	}
	DEBUG_PRINT("You're listening to noise/low amplitude sounds :( \r\n");
	return 0;
}
int IsADCFinished(void) {
	return (rem_n_bufs == 0);
}

static void StopADCAcq() {
	HAL_ADC_Stop_DMA(&hadc1);
}

static void print_spectrogram(void) {
#if (DEBUGP == 1)
	start_cycle_count();
	DEBUG_PRINT("Acquisition complete, sending the following FVs\r\n");
	for(unsigned int j=0; j < N_MELVECS; j++) {
		DEBUG_PRINT("FV #%u:\t", j+1);
		for(unsigned int i=0; i < MELVEC_LENGTH; i++) {
			DEBUG_PRINT("%.2f, ", q15_to_float(mel_vectors[j][i]));
		}
		DEBUG_PRINT("\r\n");
	}
	stop_cycle_count("Print FV");
#endif
}

static void print_encoded_packet(uint8_t *packet) {
#if (DEBUGP == 1)
	char hex_encoded_packet[2*PACKET_LENGTH+1];
	hex_encode(hex_encoded_packet, packet, PACKET_LENGTH);
	DEBUG_PRINT("DF:HEX:%s\r\n", hex_encoded_packet);
#endif
}

static void encode_packet(uint8_t *packet, uint32_t* packet_cnt) {
	// BE encoding of each mel coef
	for (size_t i=0; i<N_MELVECS; i++) {
		for (size_t j=0; j<MELVEC_LENGTH; j++) {
			(packet+PACKET_HEADER_LENGTH)[(i*MELVEC_LENGTH+j)*2]   = mel_vectors[i][j] >> 8;
			(packet+PACKET_HEADER_LENGTH)[(i*MELVEC_LENGTH+j)*2+1] = mel_vectors[i][j] & 0xFF;
		}
	}
	// Write header and tag into the packet.
	make_packet(packet, PAYLOAD_LENGTH, 0, *packet_cnt);
	*packet_cnt += 1;
	if (*packet_cnt == 0) {
		// Should not happen as packet_cnt is 32-bit and we send at most 1 packet per second.
		DEBUG_PRINT("Packet counter overflow.\r\n");
		Error_Handler();
	}
}

static void send_spectrogram() {
	uint8_t packet[PACKET_LENGTH];

//	start_cycle_count();
	encode_packet(packet, &packet_cnt);
//	stop_cycle_count("Encode packet");

//	start_cycle_count();
	S2LP_Send(packet, PACKET_LENGTH);
//	stop_cycle_count("Send packet");

//	print_encoded_packet(packet);
}

static void ADC_Callback(int buf_cplt) {
	if (rem_n_bufs != -1) {
		rem_n_bufs--;
	}
	if (rem_n_bufs == 0) {
		StopADCAcq(); // stop si on a fini d'acquérir les 20 buffers
	} else if (ADCDataRdy[1-buf_cplt]) {
		DEBUG_PRINT("Error: ADC Data buffer full\r\n");
		Error_Handler(); // erreur si l'autre moitié n'a pas été
		// traitée à temps
	}
	// traitement des données si le demi buffer est complet et que le power_threshold est atteint
	if (ADCDataRdy[buf_cplt] == 0 && power_threshold_reached()) {
		ADCDataRdy[buf_cplt] = 1;
		Spectrogram_Format((q15_t *)ADCData[buf_cplt]);
		Spectrogram_Compute((q15_t *)ADCData[buf_cplt], mel_vectors[cur_melvec]);
		cur_melvec++;
	}
	ADCDataRdy[buf_cplt] = 0; // Marque la moitié du buffer comme traitée

	if (rem_n_bufs == 0) {
//		print_spectrogram();
		send_spectrogram(); // envoie les données qd on a fini d'acquérir les 20 buffers
	}
}

void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc)
{
	ADC_Callback(1);
	// ADC_Callback(1) est appelé lorsque la conversion est complète
	// (tous les échantillons ont été convertis)
}

void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef *hadc)
{
	// ADC_Callback(0) est appelé lorsque la moitié des échantillons a été convertie
	// pour traiter les données du demi buffer
	ADC_Callback(0);
}
