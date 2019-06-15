import json
import sys
import os
import psutil
import numpy as np
import sounddevice as sd
import queue
import matplotlib.pyplot as plt

from keras import models
from utils_analysis import stft, melspectrogram, power_to_db


# CONFIGS #####################################################################
# load keras model
model = models.load_model(os.path.dirname(os.path.abspath(__file__)) + '/model/clf.h5')

# load config file
with open(os.path.dirname(os.path.abspath(__file__)) + '/model/cfg.json') as data_file:
    cfg = json.load(data_file)

# load labels
with open(os.path.dirname(os.path.abspath(__file__)) + '/model/labels.json') as data_file:
    labels = json.load(data_file)


# SETUP PLOTTING ##############################################################
fig, ax = plt.subplots()
fig.set_size_inches(12, 9)
idx = np.arange(1, len(labels) + 1)
idx_max = 0
plt.show(block=False)
plt.box(False)
ax.xaxis.grid(True)
ax.set_yticks(idx)
ax.set_yticklabels(labels)
ax.set_xlim([0, 1])
bars = plt.barh(idx, [0.5]*len(labels))
for bar in bars:
    bar.set_facecolor('#cccccc')


# INITIALIZE RECORD ###########################################################
# parameters
loopCount = 0
blockSize = 4 # predict each X frames (sliding prediction)
frameSize = cfg['hop_length'] # framesize of fft
sd.default.samplerate = cfg['target_sr'] # define sampling rate
print(sd.query_devices(device=None, kind=None))

# initialize queue, audio list
q = queue.Queue()
audio = []

# initialize spectrum ringbuffer
spec_buffer = np.zeros((cfg['n_freq_bins'], cfg['n_time_bins']))

# average
cfg['averaging'] = 2
prob_avg = np.zeros((len(labels), cfg['averaging']))
i = 0

# recording callback
def callback(indata, frames, time, status):
    if any(indata):
        q.put(indata.copy())
    else:
        print('no input')


# PREDICT #####################################################################
with sd.InputStream(device=None, channels=1, callback=callback,
                    blocksize=frameSize, samplerate=cfg['target_sr']):
    while True:
        audio.append(q.get())

        if len(audio) is blockSize:
            q.queue.clear()
            loopCount += 1

            # save audio to array for further processing
            audio_arr = np.squeeze(np.array([item for sublist in audio for item in sublist])).astype(np.float64)

            # reset audio list
            audio = []

            # calculate specs
            spec = stft(y=audio_arr,
                        n_fft=cfg['n_fft'],
                        win_length=cfg['win_length'],
                        hop_length=cfg['hop_length'],
                        center=cfg['center_win'])

            # transform to dB scale
            spec = power_to_db(
                        melspectrogram(
                            S=np.abs(spec)**2,
                            sr=cfg['target_sr'],
                            n_mels=cfg['n_freq_bins'],
                            fmin=cfg['fmin'],
                            fmax=cfg['fmax'],
                            htk=True)
                            )

            # write specs in spectrum ringbuffer
            spec_buffer[:, 0:spec.shape[1]] = spec
            spec_buffer = np.roll(spec_buffer, -spec.shape[1])

            # scale & reshape
            spec_buffer_reshape = spec_buffer.reshape(1, cfg['n_freq_bins'], cfg['n_time_bins'], 1)

            # predict
            prob = np.squeeze(model.predict(x=spec_buffer_reshape))

            # stdout to monitor metrics
            print('prediction no.: ' + str(loopCount))
            print('cpu: ' + str(psutil.cpu_percent(interval=None)))
            print('prob: ' + str(prob))
            print('labels: ' + str(labels) + '\n')

            # moving average of predictions
            prob_avg[:, 0] = prob
            prob_avg = np.roll(prob_avg, -1, axis=1)
            prob_mean = np.squeeze(np.mean(prob_avg, axis=1))

            # set maximum value graph to red
            idx_max = np.argmax(prob_mean)
            bars[idx_max].set_facecolor('#c72822')

            # update bar values
            for i, val in enumerate(prob_mean):
                bars[i].set_width(val)
            fig.canvas.draw_idle()

            try:
                fig.canvas.flush_events()

            except NotImplementedError:
                pass

            # reset colors to grey to update bars
            bars[idx_max].set_facecolor('#cccccc')
