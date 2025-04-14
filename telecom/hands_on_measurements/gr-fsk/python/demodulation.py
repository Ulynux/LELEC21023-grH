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


"""
    
def viterbi_decoder(x_tilde):

    R1 = np.array([2,1,3,0])
    R0 = np.array([0,3,1,2])
    out_R1 = np.array([[1,1],[1,0],[1,1],[1,0]])
    out_R0 = np.array([[0,0],[0,1],[0,0],[0,1]])
    symb_R1 = np.array([1.0 + 1.0j, 1.0 + 0.0j, 1.0 + 1.0j, 1.0 + 0.0j])
    symb_R0 = np.array([0.0 + 0.0j, 0.0 + 1.0j, 0.0 + 0.0j, 0.0 + 1.0j])
    len_b = 128

    def dist(a,b):
        distance = np.abs(a-b)
        # distance = np.abs(np.real(a)-np.real(b)) + np.abs(np.imag(a)-np.imag(b))
        return distance
    
    # Reshape of the received sequence to have u and c
    u_hat = x_tilde[:len(x_tilde)//2]
    c_hat = x_tilde[len(x_tilde)//2:]
    x_tilde = u_hat + 1j*c_hat
    
    N_b = int(len(x_tilde)/len_b)
    print("shape ",x_tilde.shape)
    
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
"""

def demodulate(y, B, R, Fdev):
    """
    Non-coherent demodulator.
    """
    # Receiver oversampling factor
    nb_syms = len(y) // R  # Number of CPFSK symbols in y

    # Group symbols together, in a matrix. Each row contains the R samples over one symbol period
    y = np.resize(y, (nb_syms, R))

    # Generate the reference waveforms used for the correlation
    pih = 2 * np.pi * Fdev * (np.arange(R) / R) / B
    ref_wave_1 = np.exp(1j * pih)
    ref_wave_0 = np.exp(-1j * pih)

    # Compute the correlations with the two reference waveforms
    r0 = np.abs(np.sum(y * np.conj(ref_wave_0), axis=1))
    r1 = np.abs(np.sum(y * np.conj(ref_wave_1), axis=1))

    # Perform the decision based on r0 and r1
    bits_hat = np.where(r0 > r1, 0, 1)

    #bits_hat = viterbi_decoder(bits_hat)

    return bits_hat


class demodulation(gr.basic_block):
    """
    docstring for block demodulation
    """

    def __init__(self, drate, fdev, fsamp, payload_len, crc_len):
        self.drate = drate
        self.fdev = fdev
        self.fsamp = fsamp
        self.frame_len = payload_len + crc_len
        self.osr = int(fsamp / drate)

        gr.basic_block.__init__(
            self, name="Demodulation", in_sig=[np.complex64], out_sig=[np.uint8]
        )

        self.gr_version = gr.version()

        # Redefine function based on version
        if LooseVersion(self.gr_version) < LooseVersion("3.9.0"):
            print("Compiling the Python codes for GNU Radio 3.8")
            self.forecast = self.forecast_v38
        else:
            print("Compiling the Python codes for GNU Radio 3.10")
            self.forecast = self.forecast_v310

    def forecast_v38(self, noutput_items, ninput_items_required):
        """
        input items are samples (with oversampling factor)
        output items are bytes
        """
        ninput_items_required[0] = noutput_items * self.osr * 8

    def forecast_v310(self, noutput_items, ninputs):
        """
        forecast is only called from a general block
        this is the default implementation
        """
        ninput_items_required = [0] * ninputs
        for i in range(ninputs):
            ninput_items_required[i] = noutput_items * self.osr * 8

        return ninput_items_required

    def symbols_to_bytes(self, symbols):
        """
        Converts symbols (bits here) to bytes
        """
        if len(symbols) == 0:
            return []

        n_bytes = int(len(symbols) / 8)
        bitlists = np.array_split(symbols, n_bytes)
        out = np.zeros(n_bytes).astype(np.uint8)

        for i, l in enumerate(bitlists):
            for bit in l:
                out[i] = (out[i] << 1) | bit

        return out

    def general_work(self, input_items, output_items):
        n_syms = len(output_items[0]) * 8
        buf_len = n_syms * self.osr

        y = input_items[0][:buf_len]
        self.consume_each(buf_len)

        s = demodulate(y, self.drate, self.osr, self.fdev)
        b = self.symbols_to_bytes(s)
        output_items[0][: len(b)] = b

        return len(b)

