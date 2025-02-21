from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

BIT_RATE = 50e3
PREAMBLE = np.array([int(bit) for bit in f"{0xAAAAAAAA:0>32b}"])
SYNC_WORD = np.array([int(bit) for bit in f"{0x3E2A54B7:0>32b}"])


class Chain:
    name: str = ""

    # Communication parameters
    bit_rate: float = BIT_RATE
    freq_dev: float = BIT_RATE / 4

    osr_tx: int = 64
    osr_rx: int = 8

    preamble: np.ndarray = PREAMBLE
    sync_word: np.ndarray = SYNC_WORD

    payload_len: int = 800  # Number of bits per packet

    # Simulation parameters
    n_packets: int = 100  # Number of sent packets

    # Channel parameters
    sto_val: float = 0
    sto_range: float = 10 / BIT_RATE  # defines the delay range when random

    cfo_val: float = 0
    cfo_range: float = (
        1000  # defines the CFO range when random (in Hz) #(1000 in old repo)
    )

    snr_range: np.ndarray = np.arange(-10, 25)

    # Lowpass filter parameters
    numtaps: int = 100
    #cutoff: float = BIT_RATE * osr_rx / 2.0001  # or 2*BIT_RATE,...
    cutoff = 130000

    # Tx methods

    def modulate(self, bits: np.array) -> np.array:
        """
        Modulates a stream of bits of size N
        with a given TX oversampling factor R (osr_tx).

        Uses Continuous-Phase FSK modulation.

        :param bits: The bit stream, (N,).
        :return: The modulates bit sequence, (N * R,).
        """
        fd = self.freq_dev  # Frequency deviation, Delta_f
        B = self.bit_rate  # B=1/T
        h = 2 * fd / B  # Modulation index
        R = self.osr_tx  # Oversampling factor

        x = np.zeros(len(bits) * R, dtype=np.complex64)
        ph = 2 * np.pi * fd * (np.arange(R) / R) / B  # Phase of reference waveform

        phase_shifts = np.zeros(
            len(bits) + 1
        )  # To store all phase shifts between symbols
        phase_shifts[0] = 0  # Initial phase

        for i, b in enumerate(bits):
            x[i * R : (i + 1) * R] = np.exp(1j * phase_shifts[i]) * np.exp(
                1j * (1 if b else -1) * ph
            )  # Sent waveforms, with starting phase coming from previous symbol
            phase_shifts[i + 1] = phase_shifts[i] + h * np.pi * (
                1 if b else -1
            )  # Update phase to start with for next symbol

        return x

    # Rx methods
    bypass_preamble_detect: bool = True

    def preamble_detect(self, y: np.array) -> Optional[int]:
        """
        Detects the preamlbe in a given received signal.

        :param y: The received signal, (N * R,).
        :return: The index where the preamble starts,
            or None if not found.
        """
        raise NotImplementedError

    bypass_cfo_estimation: bool = True

    def cfo_estimation(self, y: np.array) -> float:
        """
        Estimates the CFO based on the received signal.

        :param y: The received signal, (N * R,).
        :return: The estimated CFO.
        """
        raise NotImplementedError

    bypass_sto_estimation: bool = True

    def sto_estimation(self, y: np.array) -> float:
        """
        Estimates the STO based on the received signal.

        :param y: The received signal, (N * R,).
        :return: The estimated STO.
        """
        raise NotImplementedError

    def demodulate(self, y: np.array) -> np.array:
        """
        Demodulates the received signal.

        :param y: The received signal, (N * R,).
        :return: The signal, after demodulation.
        """
        raise NotImplementedError


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

    bypass_cfo_estimation = False

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
        w = 15
        p = 1
        y_smoothed = savgol_filter(phase_function, w, p)
        plt.plot(y_smoothed[:100])
        plt.plot(phase_function[:100])
        plt.title("Signal lissé")
        plt.show()



        phase_derivative_1 = phase_function[1:] - phase_function[:-1]
        phase_derivative_2 = np.abs(phase_derivative_1[1:] - phase_derivative_1[:-1])

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
