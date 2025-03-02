from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from chain import Chain

BIT_RATE = 50e3
PREAMBLE = np.array([int(bit) for bit in f"{0xAAAAAAAA:0>32b}"])
SYNC_WORD = np.array([int(bit) for bit in f"{0x3E2A54B7:0>32b}"])






class BasicChain(Chain):
    name = "Basic Tx/Rx chain"

    cfo_val, sto_val = np.nan, np.nan  # CFO and STO are random
    
    bypass_preamble_detect = False

    def preamble_detect(self, y):
        """
        Detect a preamble computing the received energy (average on a window).
        """
        L = 4 * self.osr_rx
        y_abs = np.abs(y)

        for i in range(0, int(len(y) / L)):
            sum_abs = np.sum(y_abs[i * L : (i + 1) * L])
            if sum_abs > (L - 1):  # fix threshold
                return i * L

        return None

    bypass_cfo_estimation = True

    def cfo_estimation(self, y):
        """
        Estimates CFO using Moose algorithm, on first samples of preamble.
        """
        R = self.osr_rx # Receiver oversampling factor
        
        N = 64 # Block of 4 bits (instructions) / Block of 2 bits (respect condition |cfo| < B/2N with cfo_range = 10e4)
        Nt = N*R # Number of blocks used for CFO estimation
        T = 1/self.bit_rate  # B=1/T
        
        # Extract 2 blocks of size N*R at the start of y
        block1 = y[:Nt]
        block2 = y[Nt:2*Nt] 
        alpha_hat = np.sum(block2 * np.conj(block1))

        # Apply the Moose algorithm on these two blocks to estimate the CFO
        cfo_est = (1/(2*np.pi*T*Nt/R)) * np.angle(alpha_hat)

        return cfo_est

    bypass_sto_estimation = False

    def sto_estimation(self, y):
        """
        Estimates symbol timing (fractional) based on phase shifts.
        """
        R = self.osr_rx  # Receiver oversampling factor

        # Computation of derivatives of phase function
        phase_function = np.unwrap(np.angle(y))

        # Smooth the phase function using Savitzky-Golay filter
        phase_function_smooth = savgol_filter(phase_function, window_length=15, polyorder=2)

        

        phase_derivative_1 = phase_function_smooth[1:] - phase_function_smooth[:-1]
        p_d = phase_function_smooth[1:] - phase_function_smooth[:-1]

        phase_derivative_2 = np.abs(phase_derivative_1[1:] - phase_derivative_1[:-1])
        p_d_2 = np.abs(p_d[1:]) - p_d[:-1]





        sum_der_saved = -np.inf
        save_i = 0
        for i in range(0, R):
            sum_der = np.sum(phase_derivative_2[i::R])  # Sum every R samples

            if sum_der > sum_der_saved:
                sum_der_saved = sum_der
                save_i = i

        return np.mod(save_i + 1, R)

    def demodulate(self, y):
        """
        Non-coherent demodulator.
        """
        R = self.osr_rx  # Receiver oversampling factor
        nb_syms = len(y) // R  # Number of CPFSK symbols in y
        fd = self.freq_dev  # Frequency deviation, Delta_f
        B = self.bit_rate  # B=1/T

        # Group symbols together, in a matrix. Each row contains the R samples over one symbol period
        y = np.resize(y, (nb_syms, R))
        
        # Generate the reference waveforms used for the correlation
        phi = 2 * np.pi * fd * (np.arange(R) / R) / B
        ref_wave_1 = np.exp(1j * phi)
        ref_wave_0 = np.exp(-1j * phi)

        bits_hat = np.zeros(nb_syms, dtype=int)  # Default value, all bits=0. 
        
        # Compute the correlations with the two reference waveforms (r0 and r1)
        for i in range(nb_syms):
            r0 = np.abs(np.sum(y[i] * np.conj(ref_wave_0)))
            r1 = np.abs(np.sum(y[i] * np.conj(ref_wave_1)))
            
            bits_hat[i] = 0 if np.abs(r0) > np.abs(r1) else 1  # Performs the decision based on r0 and r1


        return bits_hat
