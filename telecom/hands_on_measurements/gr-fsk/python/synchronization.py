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
from scipy.signal import savgol_filter
import numpy as np
import pmt
from gnuradio import gr

from .utils import logging, measurements_logger


def cfo_estimation(y, B, R, Fdev):
    """
    Estimates CFO using Moose algorithm, on first samples of preamble.
    """

    
    N = np.array([2,4,8]) # Block of 4 bits (instructions) / Block of 2 bits (respect condition |cfo| < B/2N with cfo_range = 10e4)
    Nt = N*R # Number of blocks used for CFO estimation
    T = 1/B # B=1/T

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


def sto_estimation(y, B, R, Fdev):

    """
    Estimates symbol timing (fractional) based on phase shifts.
    """

    order = 2 # Order of the finite difference
    

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

    phase_function = savgol_filter(phase_function, 51, 3)
    phase_derivative_2 = np.abs(second_derivative(phase_function, order))
    #phase_derivative_2 = np.abs(weno5_second_derivative(phase_function, 1/R))

    sum_der_saved = -np.inf
    save_i = 0
    for i in range(0, R):
        sum_der = np.sum(phase_derivative_2[i::R])  # Sum every R samples

        if sum_der > sum_der_saved:
            sum_der_saved = sum_der
            save_i = i

    # return np.mod(save_i + 1, R)
    return save_i


class synchronization(gr.basic_block):
    """
    docstring for block synchronization
    """

    def __init__(self, drate, fdev, fsamp, hdr_len, packet_len, tx_power,enable_log):
        self.drate = drate
        self.fdev = fdev
        self.fsamp = fsamp
        self.osr = int(fsamp / drate)
        self.hdr_len = hdr_len
        self.packet_len = packet_len  # in bytes
        self.estimated_noise_power = 0
        self.tx_power = tx_power
        self.enable_log = enable_log

        # Remaining number of samples in the current packet
        self.rem_samples = 0
        self.init_sto = 0
        self.cfo = 0.0
        self.t0 = 0.0
        self.power_est = None
        self.power_est = None
        self.estimated_noise_power = 0

        gr.basic_block.__init__(
            self, name="Synchronization", in_sig=[np.complex64], out_sig=[np.complex64]
        )
        self.logger = logging.getLogger("sync")
        self.message_port_register_in(pmt.intern("NoisePow"))
        self.set_msg_handler(pmt.intern("NoisePow"), self.handle_msg)

        self.gr_version = gr.version()

        # Redefine function based on version
        if LooseVersion(self.gr_version) < LooseVersion("3.9.0"):
            self.forecast = self.forecast_v38
        else:
            self.forecast = self.forecast_v310


    def forecast_v38(self, noutput_items, ninput_items_required):
        """
        input items are samples (with oversampling factor)
        output items are samples (with oversampling factor)
        """
        if self.rem_samples == 0:  # looking for a new packet
            ninput_items_required[0] = (
                    min(8000, 8 * self.osr * (self.packet_len + 1) + self.osr)
                )  # enough samples to find a header inside
        else:  # processing a previously found packet
            ninput_items_required[0] = (
                noutput_items  # pass remaining samples in packet to next block
            )

    def forecast_v310(self, noutput_items, ninputs):
        """
        forecast is only called from a general block
        this is the default implementation
        """
        ninput_items_required = [0] * ninputs
        for i in range(ninputs):
            if self.rem_samples == 0:  # looking for a new packet
                ninput_items_required[i] = (
                    min(8000, 8 * self.osr * (self.packet_len + 1) + self.osr)
                )  # enough samples to find a header inside
            else:  # processing a previously found packet
                ninput_items_required[i] = (
                    noutput_items  # pass remaining samples in packet to next block
                )

        return ninput_items_required

    def set_enable_log(self, enable_log):
        self.enable_log = enable_log

    def handle_msg(self, msg):
        self.estimated_noise_power = pmt.to_double(msg)

    def set_tx_power(self, tx_power):
        self.tx_power = tx_power

    def general_work(self, input_items, output_items):
        if self.rem_samples == 0:  # new packet to process, compute the CFO and STO
            y = input_items[0][: self.hdr_len * 8 * self.osr]
            self.cfo = cfo_estimation(y, self.drate, self.osr, self.fdev)

            # Correct CFO in preamble
            t = np.arange(len(y)) / (self.drate * self.osr)
            y_cfo = np.exp(-1j * 2 * np.pi * self.cfo * t) * y
            self.t0 = t[-1]

            sto = sto_estimation(y_cfo, self.drate, self.osr, self.fdev)

            self.init_sto = sto
            self.power_est = None
            self.rem_samples = (self.packet_len + 1) * 8 * self.osr
            if self.enable_log:
                self.logger.info(
                    f"new preamble detected @ {self.nitems_read(0) + sto} (CFO {self.cfo:.2f} Hz, STO {sto})"
                )
            measurements_logger.info(f"CFO={self.cfo},STO={sto}")
            self.consume_each(sto)  # drop *sto* samples to align the buffer
            return 0  # ... but we do not transmit data to the demodulation stage
        else:
            win_size = min(len(output_items[0]), self.rem_samples)
            y = input_items[0][:win_size]

            if self.power_est is None and win_size >= 256:
                self.power_est = np.var(y)
                SNR_est = (
                    self.power_est - self.estimated_noise_power
                ) / self.estimated_noise_power
                if self.enable_log:
                    self.logger.info(
                        f"estimated SNR: {10 * np.log10(SNR_est):.2f} dB ({len(y)} samples, Esti. RX power: {self.power_est:.2e},  TX indicative Power: {self.tx_power} dB)"
                    )
                measurements_logger.info(
                    f"SNRdB={10 * np.log10(SNR_est):.2f},TXPdB={self.tx_power}"
                )

            # Correct CFO before transferring samples to demodulation stage
            t = self.t0 + np.arange(1, len(y) + 1) / (self.drate * self.osr)
            y_corr = np.exp(-1j * 2 * np.pi * self.cfo * t) * y
            self.t0 = t[
                -1
            ]  # we keep the CFO correction continuous across buffer chunks

            output_items[0][:win_size] = y_corr

            self.rem_samples -= win_size
            if (
                self.rem_samples == 0
            ):  # Thow away the extra OSR samples from the preamble detection stage
                self.consume_each(win_size + self.osr - self.init_sto)
            else:
                self.consume_each(win_size)

            return win_size
