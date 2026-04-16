import numpy as np

from scipy.stats import skew, kurtosis

from openconmo.benchmark_methods import DRS
from openconmo.utils import bandpass_filter
from utils import HiddenPrints, squared_envelope


def preprocess_signals(dataset, methods=["DRS", "bandpass_filter", "squared_envelope"]):
    dataset_preprocessed = {}
    fs = 20480

    for k, signal in dataset.items():
        # DRS
        if "DRS" in methods:
            # Only keep the random part of the DRS, which is more informative for bearing fault diagnosis
            signal, _ = DRS(signal, 3000, 12_000)

        # Bandpass filtering
        if "bandpass_filter" in methods:
            with HiddenPrints():  # Ignore this line
                signal = bandpass_filter(
                    signal,
                    fs,
                    # NOTE: You can play with these parameters
                    3000,  # Bandpass center frequency
                    1500,  # Bandpass bandwidth
                )

        # Squared envelope
        if "squared_envelope" in methods:
            signal = squared_envelope(signal)

        dataset_preprocessed[k] = signal[fs : len(signal) - fs]

    return dataset_preprocessed


# Feature extraction methods #


def mean(samples, rpms):
    return np.mean(samples, axis=1)


def rms(samples, rpms):
    return np.sqrt(np.mean(np.square(samples), axis=1))


def variance(samples, rpms):
    return np.var(samples, axis=1)


def skewness(samples, rpms):
    return skew(samples, axis=1)


def kurtosis_(samples, rpms):
    return kurtosis(samples, axis=1)


def peak_to_peak(samples, rpms):
    return np.ptp(samples, axis=1)


def crest_factor(samples, rpms):
    return np.max(np.abs(samples), axis=1) / rms(samples, rpms)


def BPFO_1(samples, rpms):
    bpfo_indices = [125 if rpm == 937 else 68 for rpm in rpms]
    bpfo_indices_expanded = np.array([np.r_[idx - 6 : idx + 6] for idx in bpfo_indices])

    samples = np.abs(np.fft.rfft(samples, axis=1))[:, 1:]
    samples = np.take_along_axis(samples, bpfo_indices_expanded, axis=1).max(axis=1)
    return samples


def BPFO_2(samples, rpms):
    bpfo_indices = [125 * 2 if rpm == 937 else 68 * 2 for rpm in rpms]
    bpfo_indices_expanded = np.array([np.r_[idx - 6 : idx + 6] for idx in bpfo_indices])

    samples = np.abs(np.fft.rfft(samples, axis=1))[:, 1:]
    samples = np.take_along_axis(samples, bpfo_indices_expanded, axis=1).max(axis=1)
    return samples


def BPFO_3(samples, rpms):
    bpfo_indices = [125 * 3 if rpm == 937 else 68 * 3 for rpm in rpms]
    bpfo_indices_expanded = np.array([np.r_[idx - 6 : idx + 6] for idx in bpfo_indices])

    samples = np.abs(np.fft.rfft(samples, axis=1))[:, 1:]
    samples = np.take_along_axis(samples, bpfo_indices_expanded, axis=1).max(axis=1)
    return samples


def BPFI_1(samples, rpms):
    bpfi_indices = [163 if rpm == 937 else 90 for rpm in rpms]
    bpfi_indices_expanded = np.array([np.r_[idx - 6 : idx + 6] for idx in bpfi_indices])

    samples = np.abs(np.fft.rfft(samples, axis=1))[:, 1:]
    samples = np.take_along_axis(samples, bpfi_indices_expanded, axis=1).max(axis=1)
    return samples


def BPFI_2(samples, rpms):
    bpfi_indices = [163 * 2 if rpm == 937 else 90 * 2 for rpm in rpms]
    bpfi_indices_expanded = np.array([np.r_[idx - 6 : idx + 6] for idx in bpfi_indices])

    samples = np.abs(np.fft.rfft(samples, axis=1))[:, 1:]
    samples = np.take_along_axis(samples, bpfi_indices_expanded, axis=1).max(axis=1)
    return samples


def BPFI_3(samples, rpms):
    bpfi_indices = [163 * 3 if rpm == 937 else 90 * 3 for rpm in rpms]
    bpfi_indices_expanded = np.array([np.r_[idx - 6 : idx + 6] for idx in bpfi_indices])

    samples = np.abs(np.fft.rfft(samples, axis=1))[:, 1:]
    samples = np.take_along_axis(samples, bpfi_indices_expanded, axis=1).max(axis=1)
    return samples
