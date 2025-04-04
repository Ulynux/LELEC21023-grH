/*
* packet.c
*/

#include "aes_ref.h"
#include "config.h"
#include "packet.h"
#include "main.h"
#include "utils.h"
#include "aes.h"
#include "stm32l4xx_hal_cryp.h"

const uint8_t AES_Key[16] = {
0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00,
0x00,0x00,0x00,0x00};

void tag_cbc_mac(uint8_t *tag, const uint8_t *msg, size_t msg_len) {
uint8_t tag_hardware_full[816] = {0};
uint8_t msg_padded[816] = {0};
for (int p = 0; p < 808; p++) {
msg_padded[p] = msg[p];
}
HAL_CRYP_AESCBC_Encrypt(&hcryp, msg_padded, 816, tag_hardware_full, 4000000);
//printf("tag en hardware : ");
for (int o = 800; o < 816; o++) {
//printf("%X ", tag_hardware_full[o]);
tag[o-800] = tag_hardware_full[o];
}
//printf("\n");
}

// Assumes payload is already in place in the packet
int make_packet(uint8_t *packet, size_t payload_len, uint8_t sender_id, uint32_t serial) {
size_t packet_len = payload_len + PACKET_HEADER_LENGTH + PACKET_TAG_LENGTH;
// Initially, the whole packet header is set to 0s
memset(packet, 0, PACKET_HEADER_LENGTH);
// So is the tag
memset(packet + payload_len + PACKET_HEADER_LENGTH, 0, PACKET_TAG_LENGTH);

// TO DO : replace the two previous command by properly
// setting the packet header with the following structure :
/***************************************************************************
* Field Length (bytes) Encoding Description
***************************************************************************
* r 1 Reserved, set to 0.
* emitter_id 1 BE Unique id of the sensor node.
* payload_length 2 BE Length of app_data (in bytes).
* packet_serial 4 BE Unique and incrementing id of the packet.
* app_data any The feature vectors.
* tag 16 Message authentication code (MAC).
*
* Note : BE refers to Big endian
* Use the structure packet[x] = y; to set a byte of the packet buffer
* To perform bit masking of the specific bytes you want to set, you can use
* - bitshift operator (>>),
* - and operator (&) with hex value, e.g.to perform 0xFF
* This will be helpful when setting fields that are on multiple bytes.
*/

// For the tag field, you have to calculate the tag. The function call below is correct but
// tag_cbc_mac function, calculating the tag, is not implemented.

packet[0] = 0; // r
packet[1] = sender_id; // emitter_id
packet[2] = (payload_len >> 8) & 0xFF; // payload_length
packet[3] = payload_len & 0xFF; // payload_length
packet[4] = (serial >> 24) & 0xFF; // packet_serial
packet[5] = (serial >> 16) & 0xFF; // packet_serial
packet[6] = (serial >> 8) & 0xFF; // packet_serial
packet[7] = serial & 0xFF; // packet_serial
tag_cbc_mac(packet + payload_len + PACKET_HEADER_LENGTH, packet, payload_len + PACKET_HEADER_LENGTH);

return packet_len;
}
