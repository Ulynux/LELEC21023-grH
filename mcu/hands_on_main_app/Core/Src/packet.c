/*
 * packet.c
 */

#include "aes_ref.h"
#include "config.h"
#include "packet.h"
#include "main.h"
#include "utils.h"

const uint8_t AES_Key[16]  = {
                            0x00,0x00,0x00,0x00,
							0x00,0x00,0x00,0x00,
							0x00,0x00,0x00,0x00,
							0x00,0x00,0x00,0x00};

const uint8_t R1[4] = {2, 1, 3, 0};
const uint8_t R0[4] = {0, 3, 1, 2};
const uint8_t *out_R1[4][2] = {{1, 1}, {1, 0}, {1, 1}, {1, 0}};
const uint8_t *out_R0[4][2] = {{0, 0}, {0, 1}, {0, 0}, {0, 1}};


void tag_cbc_mac(uint8_t *tag, const uint8_t *msg, size_t msg_len) {
	// Allocate a buffer of the key size to store the input and result of AES
	// uint32_t[4] is 4*(32/8)= 16 bytes long
	uint32_t statew[4] = {0};
	// state is a pointer to the start of the buffer
	uint8_t *state = (uint8_t*) statew;
    size_t i;


    // TO DO : Complete the CBC-MAC_AES

    // Copy the result of CBC-MAC-AES to the tag.
    for (int j=0; j<16; j++) {
        tag[j] = state[j];
    }
}

void conv_encoder(const uint8_t *u, uint8_t *c, uint16_t len_u) {
    
	uint16_t len_b = 100;
	uint16_t nb_states = len_u / len_b;
    uint16_t N_b = len_u / len_b;
    
    // Allocate memory for block decomposition
    uint8_t *u_b[len_b];
    uint8_t *c_b[len_b];
    
    // Block convolutional encoding
    for (int i = 0; i < N_b; i++) {
		// Copy one block of u to u_b
		for (int j = 0; j < len_b; j++) {
			u_b[j] = u[i * len_b + j];
		}
        int state = 0;
        for (int j = 0; j < len_b; j++) {
            int index = i * len_b + j;
            if (u_b[index] == 1) {
                c_b[index] = out_R1[state][1];
                state = R1[state];
            } else {
                c_b[index] = out_R0[state][1];
                state = R0[state];
            }
        }
		// Copy result to output
		for (int j = 0; j < len_u; j++) {
			c[i * len_b + j] = c_b[j];
		}
    }        
    
}

// Assumes payload is already in place in the packet
int make_packet(uint8_t *packet, size_t payload_len, uint8_t sender_id, uint32_t serial) {
    size_t packet_len = payload_len + PACKET_HEADER_LENGTH + PACKET_TAG_LENGTH;
    // Initially, the whole packet header is set to 0s
    memset(packet, 0, PACKET_HEADER_LENGTH);
    // So is the tag
	memset(packet + payload_len + PACKET_HEADER_LENGTH, 0, PACKET_TAG_LENGTH);

	// TO DO :  replace the two previous command by properly
	//			setting the packet header with the following structure :
	/***************************************************************************
	 *    Field       	Length (bytes)      Encoding        Description
	 ***************************************************************************
	 *  r 					1 								Reserved, set to 0.
	 * 	emitter_id 			1 					BE 			Unique id of the sensor node.
	 *	payload_length 		2 					BE 			Length of app_data (in bytes).
	 *	packet_serial 		4 					BE 			Unique and incrementing id of the packet.
	 *	app_data 			any 							The feature vectors.
	 *	tag 				16 								Message authentication code (MAC).
	 *
	 *	Note : BE refers to Big endian
	 *		 	Use the structure 	packet[x] = y; 	to set a byte of the packet buffer
	 *		 	To perform bit masking of the specific bytes you want to set, you can use
	 *		 		- bitshift operator (>>),
	 *		 		- and operator (&) with hex value, e.g.to perform 0xFF
	 *		 	This will be helpful when setting fields that are on multiple bytes.
	*/

	// For the tag field, you have to calculate the tag. The function call below is correct but
	// tag_cbc_mac function, calculating the tag, is not implemented.
    tag_cbc_mac(packet + payload_len + PACKET_HEADER_LENGTH, packet, payload_len + PACKET_HEADER_LENGTH);

    return packet_len;
}
