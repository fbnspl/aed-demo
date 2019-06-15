import numpy as np
import sys
import io
import json
import warnings
import os
import scipy as sp
import six
from scipy import signal
from numpy.lib.stride_tricks import as_strided
from pathlib import Path
# from keras import backend as K
# from keras import models


# CONFIGS #####################################################################

# # load keras model
# model = models.load_model(os.path.dirname(os.path.abspath(__file__)) + '/model/clf.h5')

# load config file
with open(os.path.dirname(os.path.abspath(__file__)) + '/model/cfg.json') as data_file:
    cfg = json.load(data_file)

# load labels
with open(os.path.dirname(os.path.abspath(__file__)) + '/model/labels.json') as data_file:
    labels = json.load(data_file)


# FUNCTIONS ###################################################################


def stft(y, n_fft=2048, hop_length=512, win_length=None, window='hann',
         center=False, dtype=np.complex64, pad_mode='reflect'):

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # # Set the default hop, if it's not already specified
    # if hop_length is None:
    #     hop_length = int(win_length // 4)

    # fft_window = get_window(window, win_length, fftbins=True)
    fft_window = signal.get_window(window, win_length, fftbins=True)

    # # Pad the window out to n_fft size
    # fft_window = util.pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # # Pad the time series so that frames are centered
    # if center:
    #     util.valid_audio(y)
    #     y = np.pad(y, int(n_fft // 2), mode=pad_mode)


    # Window the time series.
    y_frames = frame(y, frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                           dtype=dtype,
                           order='F')

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = int(2**8 * 2**10 / (stft_matrix.shape[0] *
                                          stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        # RFFT and Conjugate here to match phase from DPWE code
        stft_matrix[:, bl_s:bl_t] = sp.fftpack.fft(fft_window *
                                            y_frames[:, bl_s:bl_t],
                                            axis=0)[:stft_matrix.shape[0]].conj()

    return stft_matrix


def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):

    S = np.asarray(S)

    if amin <= 0:
        raise ParameterError('amin must be strictly positive')

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn('power_to_db was called on complex input so phase '
                      'information will be discarded. To suppress this warning, '
                      'call power_to_db(np.abs(D)**2) instead.')
        magnitude = np.abs(S)
    else:
        magnitude = S

    if six.callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise ParameterError('top_db must be non-negative')
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec



def mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False,
        norm=1):


    if fmax is None:
        fmax = float(sr) / 2

    if norm is not None and norm != 1 and norm != np.inf:
        raise ParameterError('Unsupported norm: {}'.format(repr(norm)))

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)))
    # print(weights.shape)

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)
    # print(fftfreqs.shape)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)
    # print(mel_f.shape)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    # print(fdiff.shape)
    # print(ramps.shape)
    # print(n_mels)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i+2] / fdiff[i+1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == 1:
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2:n_mels+2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        warnings.warn('Empty filters detected in mel frequency basis. '
                      'Some channels will produce empty responses. '
                      'Try increasing your sampling rate (and fmax) or '
                      'reducing n_mels.')

    return weights


def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels, htk=htk)


def fft_frequencies(sr=22050, n_fft=2048):

    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_fft//2),
                       endpoint=True)


def hz_to_mel(frequencies, htk=False):

    frequencies = np.atleast_1d(frequencies)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    log_t = (frequencies >= min_log_hz)
    mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep

    return mels


def mel_to_hz(mels, htk=False):

    mels = np.atleast_1d(mels)

    if htk:
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region
    log_t = (mels >= min_log_mel)

    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

    return freqs


def frame(y, frame_length=2048, hop_length=512):

    # Compute the number of frames that will fit. The end may get truncated.
    n_frames = 1 + int((len(y) - frame_length) / hop_length)

    # Vertical stride is one sample
    # Horizontal stride is `hop_length` samples
    y_frames = np.lib.stride_tricks.as_strided(y, shape=(frame_length, n_frames),
                          strides=(y.itemsize, hop_length * y.itemsize))
    return y_frames


def melspectrogram(y=None, sr=22050, S=None, n_fft=2048, hop_length=512,
                   power=2.0, **kwargs):

        if S is not None:
            # Infer n_fft from spectrogram shape
            n_fft = 2 * (S.shape[0] - 1)
        else:
            # Otherwise, compute a magnitude spectrogram from input
            S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length))**power

        # Build a Mel filter
        mel_basis = mel(sr, n_fft, **kwargs)

        return np.dot(mel_basis, S)


# def audio_to_prediction(audio_arr):
#
#     # calculate specs
#     spec = stft(y=audio_arr,
#                 n_fft=cfg['n_fft'],
#                 win_length=cfg['win_length'],
#                 hop_length=cfg['win_length'],
#                 center=cfg['center_win'])
#
#     spec = power_to_db(
#                 melspectrogram(
#                     S=np.abs(spec)**2,
#                     sr=cfg['target_sr'],
#                     n_mels=cfg['n_bins'],
#                     fmin=cfg['fmin'],
#                     fmax=cfg['fmax'],
#                     htk=True)
#                     )
#
#     # write specs in ringbuffer
#     features[:, 0:cfg['blockSize']] = spec
#     features = np.roll(features, -cfg['blockSize'])
#
#     # scale & reshape
#     featuresScale = (features - features.flatten().mean()) / features.flatten().std()
#     featuresScale = featuresScale.reshape(1, cfg['n_bins'], cfg['n_frames'], 1)
#
#     # predict
#     prob = np.squeeze(model.predict(x=featuresScale))
#
#     return(prob)
