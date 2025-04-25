#!/usr/bin/env python
#
# Copyright 2021 UCLouvain.
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#

from distutils.version import LooseVersion

import numpy as np
from gnuradio import gr

from .utils import logging, measurements_logger


def reflect_data(x, width):
    # See: https://stackoverflow.com/a/20918545
    if width == 8:
        x = ((x & 0x55) << 1) | ((x & 0xAA) >> 1)
        x = ((x & 0x33) << 2) | ((x & 0xCC) >> 2)
        x = ((x & 0x0F) << 4) | ((x & 0xF0) >> 4)
    elif width == 16:
        x = ((x & 0x5555) << 1) | ((x & 0xAAAA) >> 1)
        x = ((x & 0x3333) << 2) | ((x & 0xCCCC) >> 2)
        x = ((x & 0x0F0F) << 4) | ((x & 0xF0F0) >> 4)
        x = ((x & 0x00FF) << 8) | ((x & 0xFF00) >> 8)
    elif width == 32:
        x = ((x & 0x55555555) << 1) | ((x & 0xAAAAAAAA) >> 1)
        x = ((x & 0x33333333) << 2) | ((x & 0xCCCCCCCC) >> 2)
        x = ((x & 0x0F0F0F0F) << 4) | ((x & 0xF0F0F0F0) >> 4)
        x = ((x & 0x00FF00FF) << 8) | ((x & 0xFF00FF00) >> 8)
        x = ((x & 0x0000FFFF) << 16) | ((x & 0xFFFF0000) >> 16)
    else:
        raise ValueError("Unsupported width")
    return x


def crc_poly(data, n, poly, crc=0, ref_in=False, ref_out=False, xor_out=0):
    # See : https://gist.github.com/Lauszus/6c787a3bc26fea6e842dfb8296ebd630
    g = 1 << n | poly  # Generator polynomial

    # Loop over the data
    for d in data:
        # Reverse the input byte if the flag is true
        if ref_in:
            d = reflect_data(d, 8)

        # XOR the top byte in the CRC with the input byte
        crc ^= d << (n - 8)

        # Loop over all the bits in the byte
        for _ in range(8):
            # Start by shifting the CRC, so we can check for the top bit
            crc <<= 1

            # XOR the CRC if the top bit is 1
            if crc & (1 << n):
                crc ^= g

    # Reverse the output if the flag is true
    if ref_out:
        crc = reflect_data(crc, n)

    # Return the CRC value
    return crc ^ xor_out

def viterbi_decoder(x_tilde):



    R1 = np.array([2,1,3,0])
    R0 = np.array([0,3,1,2])
    out_R1 = np.array([[1,1],[1,0],[1,1],[1,0]])
    out_R0 = np.array([[0,0],[0,1],[0,0],[0,1]])
    symb_R1 = np.array([1.0 + 1.0j, 1.0 + 0.0j, 1.0 + 1.0j, 1.0 + 0.0j])
    symb_R0 = np.array([0.0 + 0.0j, 0.0 + 1.0j, 0.0 + 0.0j, 0.0 + 1.0j])
    len_b = 1648

    def dist(a,b):
        distance = np.abs(a-b)
        # distance = np.abs(np.real(a)-np.real(b)) + np.abs(np.imag(a)-np.imag(b))
        return distance
    
    # Reshape of the received sequence to have u and c
    u_hat = x_tilde[:len(x_tilde)//2]
    c_hat = x_tilde[len(x_tilde)//2:]
    x_tilde = u_hat + 1j*c_hat
    
    N_b = int(len(x_tilde)/len_b)
    
    
    x_tilde_b = np.reshape(x_tilde,(N_b,len_b))
    u_hat_b = np.zeros(x_tilde_b.shape,dtype=np.int32)
    
    nb_states = len(R1)

    for i in range(N_b):           
        x_tilde_i  = x_tilde_b[i,:]
        u_hat_i = u_hat_b[i,:]
        
        bits = np.zeros((nb_states,len_b))
        weights = np.inf*np.ones((nb_states,))
        weights[0] = 0
        
        new_states = np.zeros((2,nb_states))
        new_weights = np.zeros((2,nb_states))
        new_bits = np.zeros((2,nb_states,len_b))  
        
        for j in range(len_b):
            for k in range(nb_states):
                new_states[1,k] = R1[k]
                new_states[0,k] = R0[k]
                new_weights[1,k] = weights[k] + dist(x_tilde_i[j],symb_R1[k])
                new_weights[0,k] = weights[k] + dist(x_tilde_i[j],symb_R0[k])       
                new_bits[1,k,:] = bits[k,:]
                new_bits[0,k,:] = bits[k,:]
                new_bits[1,k,j] = 1
                
            for k in range(nb_states):
                idx_0_filled = False
                for l in range(nb_states):
                    if new_states[0,l] == k:
                        if idx_0_filled:
                            idx_10 = 0
                            idx_11 = l
                        else:
                            idx_00 = 0
                            idx_01 = l 
                            idx_0_filled = True
                            
                    if new_states[1,l] == k:
                        if idx_0_filled:
                            idx_10 = 1
                            idx_11 = l
                        else:
                            idx_00 = 1
                            idx_01 = l 
                            idx_0_filled = True
                
                if new_weights[idx_00,idx_01] <= new_weights[idx_10,idx_11]:
                    weights[k] = new_weights[idx_00,idx_01]
                    bits[k,:] = new_bits[idx_00,idx_01,:]
                else:
                    weights[k] = new_weights[idx_10,idx_11]
                    bits[k,:] = new_bits[idx_10,idx_11,:]

        final_weight = np.inf
        for k in range(nb_states):
            if weights[k] < final_weight:
                final_weight = weights[k]
                u_hat_i[:] = bits[k,:]
    
    u_hat = np.reshape(u_hat_b,(u_hat_b.size,))
    return u_hat


class packet_parser(gr.basic_block):
    """
    docstring for block packet_parser
    """

    def __init__(self, hdr_len, payload_len, crc_len, address,log_payload):
        self.hdr_len = hdr_len
        self.payload_len = payload_len
        self.crc_len = crc_len
        self.nb_packet = 0
        self.nb_error = 0
        self.log_payload=log_payload

        self.packet_len = self.hdr_len + self.payload_len + self.crc_len
        self.address = address

        gr.basic_block.__init__(
            self,
            name="packet_parser",
            in_sig=[np.uint8],
            out_sig=[(np.uint8, self.payload_len//2)],
        )
        self.logger = logging.getLogger("parser")

        self.gr_version = gr.version()

        # Redefine function based on version
        if LooseVersion(self.gr_version) < LooseVersion("3.9.0"):
            self.forecast = self.forecast_v38
        else:
            self.forecast = self.forecast_v310



    def forecast_v38(self, noutput_items, ninput_items_required):
        ninput_items_required[0] = self.packet_len + 1  # in bytes

    def forecast_v310(self, noutput_items, ninputs):
        """
        forecast is only called from a general block
        this is the default implementation
        """
        ninput_items_required = [0] * ninputs
        for i in range(ninputs):
            ninput_items_required[i] = self.packet_len + 1  # in bytes

        return ninput_items_required
    
    def set_log_payload(self, log_payload):
        self.log_payload = log_payload

    def general_work(self, input_items, output_items):
        # we process maximum one packet at a time
        input_bytes = input_items[0][: self.packet_len + 1]
        self.consume_each(self.packet_len + 1)

        b = np.unpackbits(input_bytes)  # bytes to bits

        # print(b)

        b_hdr = b[: self.hdr_len * 8]
        v = np.abs(
            np.correlate(b_hdr * 2 - 1, np.array(self.address) * 2 - 1, mode="full")
        )
        i = np.argmax(v) + 1

        b_pkt = b[i : i + (self.payload_len + self.crc_len) * 8]

        pkt_bytes = np.packbits(b_pkt)

        payload_init = pkt_bytes[0 : self.payload_len]

        payload_bits = np.unpackbits(payload_init)
        payload_bits = viterbi_decoder(payload_bits)
        payload = np.packbits(payload_bits)

        print("P2 ",payload[2])
        print("P3 ",payload[3])

        crc = pkt_bytes[self.payload_len : self.payload_len + self.crc_len]
        print("pkt_bytes", len(pkt_bytes))
        output_items[0][0] = payload
        print("Output length", len(output_items))
        print("Output[0]", len(output_items[0]))
        print("Output[0][0]", len(output_items[0][0]))

        crc_verif = crc_poly(
            bytearray(payload_init),
            8,
            0x07,
            crc=0xFF,
            ref_in=False,
            ref_out=False,
            xor_out=0,
        )
        self.nb_packet += 1
        is_correct = all(crc == crc_verif)
        measurements_logger.info(
            f"packet_number={self.nb_packet},correct={is_correct},payload=[{','.join(map(str, payload))}]"
        )
        if is_correct:
            if self.log_payload:
                self.logger.info(f"packet successfully demodulated: {payload} (CRC: {crc})")
            output_items[0][: self.payload_len//2] = payload
            self.logger.info(
                f"{self.nb_packet} packets received with {self.nb_error} error(s)"
            )
            return 1
        else:
            if self.log_payload:
                self.logger.error(f"incorrect CRC, packet dropped: {payload} (CRC: {crc})")
            self.nb_error += 1
            self.logger.info(
                f"{self.nb_packet} packets received with {self.nb_error} error(s)"
            )
            return 0
