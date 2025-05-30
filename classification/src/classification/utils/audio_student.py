import random
from typing import Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf
import scipy
from numpy import ndarray
from scipy.signal import fftconvolve

# -----------------------------------------------------------------------------
"""
Synthesis of the classes in :
- AudioUtil : util functions to process an audio signal.
- Feature_vector_DS : Create a dataset class for the feature vectors.
"""
# -----------------------------------------------------------------------------


class AudioUtil:
    """
    Define a new class with util functions to process an audio signal.
    """

    def open(audio_file) -> Tuple[ndarray, int]:
        """
        Load an audio file.

        :param audio_file: The path to the audio file.
        :return: The audio signal as a tuple (signal, sample_rate).
        """
        sig, sr = sf.read(audio_file)
        if sig.ndim > 1:
            sig = sig[:, 0]
        return (sig, sr)

    def play(audio):
        """
        Play an audio file.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        """
        sig, sr = audio
        sd.play(sig, sr)

    def normalize(audio, target_dB=52) -> Tuple[ndarray, int]:
        """
        Normalize the energy of the signal.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param target_dB: The target energy in dB.
        """
        sig, sr = audio
        sign = sig / np.sqrt(np.sum(np.abs(sig) ** 2))
        C = np.sqrt(10 ** (target_dB / 10))
        sign *= C
        return (sign, sr)

    def resample(audio, newsr=10200) -> Tuple[np.ndarray, int]:
        """
        Resample to target sampling frequency.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param newsr: The target sampling frequency.
        """
        sig, sr = audio

        num_samples = int(len(sig) * newsr / sr)

        resig = scipy.signal.resample(sig, num_samples)

        return (resig, newsr)

    def pad_trunc(audio, max_ms) -> Tuple[ndarray, int]:
        """
        Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param max_ms: The target length in milliseconds.
        """
        sig, sr = audio
        sig_len = len(sig)
        max_len = int(sr * max_ms / 1000)

        if sig_len > max_len:
            # Calculate the energy of the signal in sliding windows
            window_size = max_len
            energies = [
                np.sum(sig[i:i + window_size] ** 2)
                for i in range(0, sig_len - window_size + 1)
            ]

            # Find the starting position with the highest energy
            start_idx = np.argmax(energies)
            sig = sig[start_idx:start_idx + max_len]

        elif sig_len < max_len:
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = np.zeros(pad_begin_len)
            pad_end = np.zeros(pad_end_len)

            sig = np.concatenate((pad_begin, sig, pad_end))

        return (sig, sr)

    def time_shift_random(audio, shift_limit=0.3) -> Tuple[ndarray, int]:
        """
        Shifts the signal to the left or right by some percent. Values at the end are 'wrapped around' to the start of the transformed signal.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param shift_limit: The percentage (between 0.0 and 1.0) by which to circularly shift the signal.
        """
        sig, sr = audio
        sig_len = len(sig)
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (np.roll(sig, shift_amt), sr)
    
    def time_shift(audio, shift_ms: float) -> Tuple[ndarray, int]:
        """
        Shifts the signal to the left or right by a fixed duration in milliseconds.
        Values at the end are 'wrapped around' to the start of the transformed signal.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param shift_ms: The duration (in milliseconds) by which to circularly shift the signal.
                        Positive values shift to the right, negative values shift to the left.
        :return: The shifted audio signal as a tuple (signal, sample_rate).
        """
        sig, sr = audio
        sig_len = len(sig)
        
        # Convert shift duration (ms) to number of samples
        shift_amt = int((shift_ms / 1000) * sr)
        
        # Ensure the shift amount is within valid bounds
        shift_amt = shift_amt % sig_len  # Handle cases where shift_amt > sig_len or < -sig_len
        
        # Perform the circular shift
        shifted_sig = np.roll(sig, shift_amt)
        
        return (shifted_sig, sr)

    def scaling(audio, scaling_limit=5) -> Tuple[ndarray, int]:
        """
        Augment the audio signal by scaling it by a random factor.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param scaling_limit: The maximum scaling factor.
        """
        sig, sr = audio

        scaling = random.uniform(0, scaling_limit)
        
        scaled_sig = sig * scaling
        
        scaled_sig = np.clip(scaled_sig, -1, 1)
        
        scaled_sig,sr = audio
        
        return audio

    def add_noise(audio, sigma=0.05) -> Tuple[ndarray, int]:
        """
        Augment the audio signal by adding gaussian noise.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param sigma: Standard deviation of the gaussian noise.
        """
        sig, sr = audio
        desired_snr_db = 20
        signal_power = np.mean(sig**2)  # Assuming 'sig' is your signal
        noise_power = signal_power / (10**(desired_snr_db / 10))
        noise_std_dev = np.sqrt(noise_power)
        
        noise = np.random.normal(0, noise_std_dev, len(sig))
        sig += noise
        sig = np.clip(sig, -1, 1)
        sig,sr = audio
        return audio

    def echo(audio, nechos=2) -> Tuple[ndarray, int]:
        """
        Add echo to the audio signal by convolving it with an impulse response. The taps are regularly spaced in time and each is twice smaller than the previous one.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param nechos: The number of echoes.
        """
        sig, sr = audio
        sig_len = len(sig)
        echo_sig = np.zeros(sig_len)
        echo_sig[0] = 1
        echo_sig[(np.arange(nechos) / nechos * sig_len).astype(int)] = (
            1 / 2
        ) ** np.arange(nechos)

        sig = fftconvolve(sig, echo_sig, mode="full")[:sig_len]
        return (sig, sr)

    def filter(audio, filt) -> Tuple[ndarray, int]:
        """
        Filter the audio signal with a provided filter. Note the filter is given for positive frequencies only and is thus symmetrized in the function.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param filt: The filter to apply.
        """
        sig, sr = audio
        conv_sig = np.fft.fft(sig)
        sig = np.fft.ifft(conv_sig * filt).real

        return (sig, sr)

    def add_bg(
        audio, dataset, num_sources=2, max_ms=2500, amplitude_limit=0.1
    ) -> Tuple[ndarray, int]:
        """
        Adds up sounds uniformly chosen at random to audio.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param dataset: The dataset to sample from.
        :param num_sources: The number of sounds to add.
        :param max_ms: The maximum duration of the sounds to add.
        :param amplitude_limit: The maximum amplitude of the added sounds.
        """
        sig, sr = audio

        # Liste des classes de sons de fond disponibles
        classes = dataset.list_classes()

        for _ in range(num_sources):
            cls = random.choice(classes)
            audio_file = random.choice(dataset.get_class_files(cls))
            
            bg_sig, bg_sr = AudioUtil.open(audio_file)
            if bg_sr != sr:
                bg_sig, _ = AudioUtil.resample((bg_sig, bg_sr), sr)   # Rééchantillonnage
            bg_sig, _ = AudioUtil.pad_trunc((bg_sig, sr), max_ms)    # Troncature/Padding
            
            bg_sig = bg_sig / np.max(np.abs(bg_sig))  # Normalisation à [-1, 1]
            bg_amplitude = random.uniform(0, amplitude_limit) * np.max(np.abs(sig))
            bg_sig = bg_sig * bg_amplitude

            # Ajouter le bruit au signal principal
            sig = sig + bg_sig[:len(sig)]  # Assurer la compatibilité des longueurs

        sig = np.clip(sig, -1, 1)

        return sig, sr
    
    def add_background_noise(audio, SNR = 20) -> Tuple[ndarray, int]:
        """
        Adds background noise from background.wav to audio.
        
        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param amplitude_limit: The maximum amplitude of the added noise.

        :return: The audio signal with added noise as a tuple (signal, sample_rate).
        """

        sig, sr = audio
        bg_sig, _ = AudioUtil.open("src/classification/datasets/soundfiles/background.wav")

        # Energies of the signals
        sig_energy = np.mean(sig**2)
        bg_energy = np.mean(bg_sig**2)

        length = len(sig)

        # Select a random part of the background noise
        start_index = random.randint(0, len(bg_sig) - length)
        bg_sig = bg_sig[start_index:start_index + length]
        
        # Add the background noise to the signal with the desired SNR
        factor = np.sqrt(sig_energy / (bg_energy * 10**(SNR/10)))
        bg_sig = bg_sig * factor
        sig = sig + bg_sig

        sig = np.clip(sig, -1, 1)

        return sig, sr


    def specgram(audio, Nft=512, fs2=10200) -> ndarray:
        """
        Compute a Spectrogram.

        :param aud: The audio signal as a tuple (signal, sample_rate).
        :param Nft: The number of points of the FFT.
        :param fs2: The sampling frequency.
        """
        sig, sr = audio
        if sr != fs2:
            sig, _ = AudioUtil.resample(audio, fs2)
            
        stft = librosa.stft(sig, n_fft=Nft, hop_length=Nft)
        
        # stft /= float(2**8)
        
        
        return stft

    def get_hz2mel(fs2=10200, Nft=512, Nmel=20) -> ndarray:
        """
        Get the hz2mel conversion matrix.

        :param fs2: The sampling frequency.
        :param Nft: The number of points of the FFT.
        :param Nmel: The number of mel bands.
        """
        mels = librosa.filters.mel(sr=fs2, n_fft=Nft, n_mels=Nmel)
        mels = mels[:, :-1]
        mels = mels / np.max(mels)

        return mels

    def melspectrogram(audio, Nmel=20, Nft=512, fs2=10200) -> ndarray:
        """
        Generate a Melspectrogram.

        :param audio: The audio signal as a tuple (signal, sample_rate).
        :param Nmel: The number of mel bands.
        :param Nft: The number of points of the FFT.
        :param fs2: The sampling frequency.
        """
        sig, sr = audio
        if sr != fs2:
            sig, _ = AudioUtil.resample(audio, fs2)

        stft = AudioUtil.specgram((sig, fs2), Nft=Nft, fs2=fs2)
        
        mels = librosa.filters.mel(sr=fs2, n_fft=Nft, n_mels=Nmel)
        mels = mels / np.max(mels)
        melspec = np.dot(mels, np.abs(stft))
        return melspec
    def split_melspectrogram(melspec, target_width=20):
        """
        Divise un melspectrogram (50, 20) en deux melspectrograms (20, 20) en sélectionnant
        les parties avec le plus d'énergie.

        :param melspec: Le melspectrogram d'entrée de dimensions (50, 20).
        :param target_width: La largeur cible pour les sous-melspectrograms.
        :return: Deux sous-melspectrograms de dimensions (20, 20).
        """
        num_cols = melspec.shape[1]
        if num_cols < target_width:
            raise ValueError("Le melspectrogram doit avoir au moins autant de colonnes que target_width.")

        # Étape 1 : Calculer l'énergie pour chaque fenêtre de 20 colonnes
        energies = [
            np.sum(melspec[:, i:i + target_width] ** 2)
            for i in range(1,num_cols - target_width + 1)
        ]

        # Étape 2 : Trouver les indices des deux fenêtres avec le plus d'énergie
        top_indices = np.argsort(energies)[-2:]  # Les deux indices avec la plus grande énergie
        top_indices = sorted(top_indices)  # Garder l'ordre pour rester ordonné

        # Étape 3 : Extraire les deux sous-melspectrograms
        sub_melspec_1 = melspec[:, top_indices[0]:top_indices[0] + target_width]
        sub_melspec_2 = melspec[:, top_indices[1]:top_indices[1] + target_width]

        return sub_melspec_1, sub_melspec_2
    def spectro_aug_timefreq_masking(
        spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1
    ) -> ndarray:
        """
        Augment the Spectrogram by masking out some sections of it in both the frequency dimension 
        (ie. horizontal bars) and the time dimension (vertical bars) to prevent overfitting and to help the model generalise better. 
        The masked sections are replaced with the mean value.


        :param spec: The spectrogram.
        :param max_mask_pct: The maximum percentage of the spectrogram to mask out.
        :param n_freq_masks: The number of frequency masks to apply.
        :param n_time_masks: The number of time masks to apply.
        """
        Nmel, n_steps = spec.shape
        mask_value = np.mean(spec)
        aug_spec = np.copy(spec)  # avoids modifying spec

        freq_mask_param = max_mask_pct * Nmel
        for _ in range(n_freq_masks):
            height = int(np.round(random.random() * freq_mask_param))
            pos_f = np.random.randint(Nmel - height)
            aug_spec[pos_f : pos_f + height, :] = mask_value

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            width = int(np.round(random.random() * time_mask_param))
            pos_t = np.random.randint(n_steps - width)
            aug_spec[:, pos_t : pos_t + width] = mask_value

        return aug_spec


class Feature_vector_DS:
    """
    Dataset of Feature vectors.
    """

    def __init__(
        self,
        dataset,
        Nft=512,
        nmel=20,
        duration=500,
        shift_pct=0,
        normalize=False,
        data_aug=None,
        pca=None,
    ):
        self.dataset = dataset
        self.Nft = Nft
        self.nmel = nmel
        self.duration = duration  # ms
        self.sr = 10200
        self.shift_pct = shift_pct  # percentage of total
        self.normalize = normalize
        self.data_aug = data_aug
        self.data_aug_factor = 1
        if isinstance(self.data_aug, list):
            self.data_aug_factor += len(self.data_aug)
        else:
            self.data_aug = [self.data_aug]
        self.ncol = int(self.duration * self.sr / (1e3 * self.Nft))  # number of columns in melspectrogram
        self.pca = pca

    def __len__(self) -> int:
        """
        Number of items in dataset.
        """
        return len(self.dataset) * self.data_aug_factor

    def get_audiosignal(self, cls_index: Tuple[str, int]) -> Tuple[ndarray, int]:
        """
        Get temporal signal of i'th item in dataset.

        :param cls_index: Class name and index.
        """
        audio_file = self.dataset[cls_index]
        aud = AudioUtil.open(audio_file)
        aud = AudioUtil.resample(aud, self.sr)
        aud = AudioUtil.time_shift(aud, self.shift_pct)
        aud = AudioUtil.pad_trunc(aud, self.duration)
        if self.data_aug is not None:
            if "add_bg" in self.data_aug:
                aud = AudioUtil.add_bg(
                    aud,
                    self.dataset,
                    num_sources=1,
                    max_ms=self.duration,
                    amplitude_limit=0.1,
                )
            if "echo" in self.data_aug:
                aud = AudioUtil.echo(aud)
            if "noise" in self.data_aug:
                aud = AudioUtil.add_noise(aud, sigma=0.0284)
            if "scaling" in self.data_aug:
                aud = AudioUtil.scaling(aud, scaling_limit=5)
            if "background_noise" in self.data_aug:
                aud = AudioUtil.add_background_noise(aud, SNR = 20)

        # aud = AudioUtil.normalize(aud, target_dB=10)
        # aud = (aud[0] / np.max(np.abs(aud[0])), aud[1])
        return aud

    def __getitem__(self, cls_index: Tuple[str, int]) -> Tuple[ndarray, int]:
        """
        Get i'th item in dataset.

        :param cls_index: Class name and index.
        """
        aud = self.get_audiosignal(cls_index)
        sgram = AudioUtil.melspectrogram(aud, Nmel=self.nmel, Nft=self.Nft)
        if self.data_aug is not None:
            if "aug_sgram" in self.data_aug:
                sgram = AudioUtil.spectro_aug_timefreq_masking(
                    sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2
                )

        sgram_crop = sgram[:, : self.ncol]
        fv = sgram_crop.flatten()  # feature vector
        if self.normalize:
            fv /= np.linalg.norm(fv)
        if self.pca is not None:
            fv = self.pca.transform([fv])[0]
        return fv

    def display(self, cls_index: Tuple[str, int]):
        """
        Play sound and display i'th item in dataset.

        :param cls_index: Class name and index.
        """
        audio = self.get_audiosignal(cls_index)
        AudioUtil.play(audio)
        plt.figure(figsize=(6, 4))
        plt.imshow(
            AudioUtil.melspectrogram(audio, Nmel=self.nmel, Nft=self.Nft),
            cmap="jet",
            origin="lower",
            aspect="auto",
        )
        plt.colorbar()
        plt.title(audio)
        plt.title(self.dataset.__getname__(cls_index))
        plt.xlabel("Melvec number [-]")
        plt.ylabel("Melvec length [-]")
        plt.savefig("output.pdf", format="pdf")
        plt.show()

    def mod_data_aug(self, data_aug) -> None:
        """
        Modify the data augmentation options.

        :param data_aug: The new data augmentation options.
        """
        self.data_aug = data_aug
        self.data_aug_factor = 1
        if isinstance(self.data_aug, list):
            self.data_aug_factor += len(self.data_aug)
        else:
            self.data_aug = [self.data_aug]