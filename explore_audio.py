import matplotlib.pyplot as plt
import numpy as np

from koogu.data import feeder
from pathlib import Path


# explore waveforms and spectrograms
# TODO: plot n random samples for each class
train_dir = "/media/datadrive/Datasets/referencedata/islandelephants/prepared/train/"
npz_file = Path(train_dir) / Path("0mkandang_2021-10-10_15-50-12.wav.npz")
fs = 1000
spectral_settings = {"win_len": 0.256, "win_overlap_prc": 0.75, "bandwidth_clip": [15, 250], "num_mels": 64}
validation_split = 0.2
max_samples_per_class = 20000
normalize_samples = True
seed = 666


data_feeder = feeder.SpectralDataFeeder(
    data_dir=train_dir,
    fs=fs,
    spec_settings=spectral_settings,
    validation_split=validation_split,
    max_clips_per_class=max_samples_per_class,
    normalize_clips=normalize_samples,
    random_state_seed=seed,
)


# TODO: make this a separate method and allow to plot n samples per class
with np.load(npz_file) as data:
    # NOTE: labels are stored [0, 1] to indicate true or false for class_labels [elephant, background]
    sample_labels = data["labels"]
    sample_clips = data["clips"]
    sample_fs = data["fs"]
    sample_channels = data["channels"]
    sample_clip_offsets = data["clip_offsets"]

for i in range(len(sample_labels)):
    if sample_labels[i][0] == 1:
        # this is an elephant sample [elephant, background]
        waveform = sample_clips[i]
        label = sample_labels[i]

        # transform waveform to spectrogram
        spectrogram, label = data_feeder.transform(waveform, label=label, is_training=True)

        # plot waveform and spectrogram
        figure, axes = plt.subplots(1, 2)
        axes[0].plot(waveform)
        axes[0].set_title(f"Waveform")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Amplitude")
        im = axes[1].imshow(spectrogram, cmap="magma")
        plt.colorbar(im, ax=axes[1], label="Decibels")
        axes[1].set_title("Spectrogram (dB)")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Frequency")

        # TODO: collect label, audio and spectral settings and add to plot
        print(label)
        print(spectrogram.shape)

        # TODO: save figure
        plt.show()
        plt.close()

        # TODO: do we need the spectrogram to be standardized for training?
