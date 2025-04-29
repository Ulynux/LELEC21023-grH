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
    freq_dev: float = BIT_RATE / 2 * 2

    osr_tx: int = 64
    osr_rx: int = 8

    preamble: np.ndarray = PREAMBLE
    sync_word: np.ndarray = SYNC_WORD
    order = 2
    payload_len: int = 824  # Number of bits per packet

    # Simulation parameters
    n_packets: int = 1000  # Number of sent packets

    # Channel parameters
    sto_val: float = np.nan
    sto_range: float = 10 / BIT_RATE  # defines the delay range when random

    cfo_val: float = np.nan
    cfo_range: float = (
        10000  # defines the CFO range when random (in Hz) #(1000 in old repo)
    )

    snr_range: np.ndarray = np.arange(-10, 25)

    # Lowpass filter parameters
    numtaps: int = 31
    #cutoff: float = BIT_RATE * osr_rx / 2.0001  # or 2*BIT_RATE,...
    cutoff = 130000

    # Viterbi encoder parameters
    R1 = np.array([2,1,3,0])
    R0 = np.array([0,3,1,2])
    out_R1 = np.array([[1,1],[1,0],[1,1],[1,0]])
    out_R0 = np.array([[0,0],[0,1],[0,0],[0,1]])
    symb_R1 = np.array([1.0 + 1.0j, 1.0 + 0.0j, 1.0 + 1.0j, 1.0 + 0.0j])
    symb_R0 = np.array([0.0 + 0.0j, 0.0 + 1.0j, 0.0 + 0.0j, 0.0 + 1.0j])
    len_b = 206*8

    # Tx methods

    bypass_viterbi: bool = True

    def conv_encoder(self, u):
        """
        This function encodes the bit stream <u> with a convolutional encoder 
        whose trellis is described by <R1>, <R0>, <out_R1> and <out_R0>, producing 
        a bit stream <c>. The encoding process works on blocks of <len_b> bits, 
        each block being encoded seprately.
        
        In the below function, the block separation has already been handled. You
        need to fill in the numpy array <c_i> which is the non-systematic part of the 
        coded sequence corresponding to the input block <u_i>.    

        Parameters
        ----------
        u : 1D numpy array
            Input sequence.
        R1 : 1D numpy array
            Trellis decomposition - transitions if 1.
        R0 : 1D numpy array
            Trellis decomposition - transitions of 0.
        out_R1 : 2D numpy array
            Trellis decomposition - output bits corresponding to transitions with 1.
        out_R0 : 2D numpy array
            Trellis decomposition - output bits corresponding to transitions with 1.
        len_b : int
            Length of each block. We assume that N_b = len(u)/len_b is an integer!

        Returns
        -------
        u : 1D numpy array
            Systematic part of the coded sequence (i.e. the input bit stream).
        c : 1D numpy array
            Non-systematic part of the coded sequence.
        """
        # Viterbi encoder parameters
        R1 = self.R1
        R0 = self.R0
        out_R1 = self.out_R1
        out_R0 = self.out_R0
        len_b = self.len_b
    
        # number of states in the trellis
        nb_states = len(R1)
        
        ## Block decomposition for the non-systematic output
        N_b = int(len(u)/len_b)
        
        u_b = np.reshape(u,(N_b,len_b))
        c_b = np.zeros(u_b.shape,dtype=np.int32)
        
        # block convolutional encoder (non-systematic output)
        for i in range(0,N_b): 
            # input of block i
            u_i = u_b[i,:]
            # non systematic output of block i (TO FILL!)
            c_i = c_b[i,:] 
            state = 0
            for j in range(0,len_b):
                if u_i[j] == 1:
                    c_i[j] = out_R1[state,1]
                    state = R1[state]

                else:
                    c_i[j] = out_R0[state,1]
                    state = R0[state]
                              
        # non-systematic output
        c = np.reshape(c_b,u.shape)
        
        return (u,c)

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

    bypass_cfo_estimation: bool = False

    def cfo_estimation(self, y: np.array) -> float:
        """
        Estimates the CFO based on the received signal.

        :param y: The received signal, (N * R,).
        :return: The estimated CFO.
        """
        raise NotImplementedError

    bypass_sto_estimation: bool = False

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
    
    bypass_preamble_detect = True

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
        B = self.bit_rate # B=1/T
        
        N = np.array([2, 4, 8]) # Block of 4 bits (instructions) / Block of 2 bits (respect condition |cfo| < B/2N with cfo_range = 10e4)
        Nt = N*R # Number of blocks used for CFO estimation
        T = 1/self.bit_rate  # B=1/T

        y = y.copy()
        y_copy = y.copy()
        cfo_est = 0
        
        for i in range(len(N)):

            # Extract 2 blocks of size N*R at the start of y
            block1 = y[:Nt[i]]
            block2 = y[Nt[i]:2*Nt[i]] 
            alpha_hat = np.sum(block2 * np.conj(block1))

            # Apply the Moose algorithm on these two blocks to estimate the CFO
            cfo_est += (1/(2*np.pi*T*Nt[i]/R)) * np.angle(alpha_hat)

            # Correct the signal with the estimated CFO
            y = np.exp(-1j * 2 * np.pi * cfo_est * np.arange(len(y)) / (B*R)) * y_copy

        return cfo_est

    bypass_sto_estimation = False

    def sto_estimation(self, y):
        """
        Estimates symbol timing (fractional) based on phase shifts.
        """
        R = self.osr_rx  # Receiver oversampling factor
        order = self.order # Order of the finite difference
        

        # Computation of derivatives of phase function
        phase_function = np.unwrap(np.angle(y))

        phase_derivative_1 = phase_function[1:] - phase_function[:-1]
        phase_derivative_2 = np.abs(phase_derivative_1[1:] - phase_derivative_1[:-1])

        def second_derivative(function, ord):
            # Compute the central finite difference of with error term of order ord (2 or 4)
            derivative = np.zeros(len(phase_function))
            
            if ord == 2:
                # (U+h − 2U0 + U-h)/h^2
                derivative[1:-1] = (function[2:] - 2 * function[1:-1] + function[:-2])

            if ord == 4:
                # (−U2h + 16Uh − 30U0 + 16U−h − U−2h)/12h^2
                derivative[2:-2] = (-function[4:] + 16 * function[3:-1] - 30 * function[2:-2] + 16 * function[1:-3] - function[:-4])

            if ord == 6:
                # 2f(x−3h)−27f(x−2h)+270f(x−h)−490f(x)+270f(x+h)−27f(x+2h)+2f(x+3h)​/180h^2
                derivative[3:-3] = (2*function[6:] - 27*function[5:-1] + 270*function[4:-2] - 490*function[3:-3] + 270*function[2:-4] - 27*function[1:-5] + 2*function[:-6])

            if ord == 8:
                # −9f(x−4h)+128f(x−3h)−1008f(x−2h)+8064f(x−h)−14350f(x)+8064f(x+h)−1008f(x+2h)+128f(x+3h)−9f(x+4h)​/5040h^2
                derivative[4:-4] = (-9*function[8:] + 128*function[7:-1] - 1008*function[6:-2] + 8064*function[5:-3] - 14350*function[4:-4] + 8064*function[3:-5] - 1008*function[2:-6] + 128*function[1:-7] - 9*function[:-8])

            return derivative
         
        # def weno5_second_derivative(f, h):
        #     """
        #     Compute the second derivative using the WENO-5 scheme.
            
        #     Parameters:
        #     f : numpy array
        #         The function values at discrete points.
        #     h : float
        #         The uniform grid spacing.

        #     Returns:
        #     numpy array
        #         The second derivative approximation.
        #     """
        #     n = len(f)
        #     fxx = np.zeros(n)

        #     # Small epsilon to prevent division by zero
        #     eps = 1e-6  

        #     for i in range(2, n - 2):  # Ensure we don't go out of bounds
        #         # Compute smoothness indicators
        #         beta0 = (13/12) * (f[i-2] - 2*f[i-1] + f[i])**2 + (1/4) * (f[i-2] - 4*f[i-1] + 3*f[i])**2
        #         beta1 = (13/12) * (f[i-1] - 2*f[i] + f[i+1])**2 + (1/4) * (f[i-1] - f[i+1])**2
        #         beta2 = (13/12) * (f[i] - 2*f[i+1] + f[i+2])**2 + (1/4) * (3*f[i] - 4*f[i+1] + f[i+2])**2

        #         # Compute nonlinear weights
        #         alpha0 = 1 / (eps + beta0)**2
        #         alpha1 = 6 / (eps + beta1)**2
        #         alpha2 = 3 / (eps + beta2)**2
        #         w0 = alpha0 / (alpha0 + alpha1 + alpha2)
        #         w1 = alpha1 / (alpha0 + alpha1 + alpha2)
        #         w2 = alpha2 / (alpha0 + alpha1 + alpha2)

        #         # Compute second derivative using WENO-5 weighting
        #         fxx[i] = (
        #             w0 * (-f[i-2] + 16*f[i-1] - 30*f[i] + 16*f[i+1] - f[i+2]) / (12*h**2)
        #             + w1 * (-f[i-2] + 16*f[i-1] - 30*f[i] + 16*f[i+1] - f[i+2]) / (12*h**2)
        #             + w2 * (-f[i-2] + 16*f[i-1] - 30*f[i] + 16*f[i+1] - f[i+2]) / (12*h**2)
        #         )

        #     return fxx
        
        #phase_function = savgol_filter(phase_function, 51, 3)
        #phase_derivative_2 = np.abs(second_derivative(phase_function, order))
        # phase_derivative_2 = np.abs(weno5_second_derivative(phase_function, 1/R))

        sum_der_saved = -np.inf
        save_i = 0
        for i in range(0, R):
            sum_der = np.sum(phase_derivative_2[i::R])  # Sum every R samples

            if sum_der > sum_der_saved:
                sum_der_saved = sum_der
                save_i = i

        # return np.mod(save_i + 1, R)
        return save_i

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

        phi = 2 * np.pi * (fd) * (np.arange(R) / R) / B

        ref_wave_1 = np.exp(1j * phi)
        ref_wave_0 = np.exp(-1j * phi)

        bits_hat = np.zeros(nb_syms, dtype=int)  # Default value, all bits=0. 
        
        for i in range(nb_syms):

            r0 = np.abs(np.sum(y[i] * np.conj(ref_wave_0)))
            r1 = np.abs(np.sum(y[i] * np.conj(ref_wave_1)))

            if r0 > r1:
                bits_hat[i] = 0
            else:
                bits_hat[i] = 1

        return bits_hat

    bypass_viterbi = True

    def viterbi_decoder(self, x_tilde):

        # Viterbi decoder parameters
        R1 = self.R1
        R0 = self.R0
        symb_R1 = self.symb_R1
        symb_R0 = self.symb_R0
        len_b = self.len_b

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

    