import matplotlib.pyplot as plt

from koogu.data import feeder
from koogu.model import architectures
from koogu import train
from pathlib import Path


def run(
    train_dir,
    validation_split,
    max_samples_per_class,
    normalize_samples,
    audio_settings,
    spectral_settings,
    training_settings,
    model_dir,
    seed,
):
    # define data feeder that transforms (waveform to spectrogram) and loads prepared audio samples
    # TODO: the input waveforms are normalized but seems the spectrograms are not normalized before feeding to the model
    data_feeder = feeder.SpectralDataFeeder(
        data_dir=train_dir,
        fs=audio_settings["desired_fs"],
        spec_settings=spectral_settings,
        validation_split=validation_split,
        max_clips_per_class=max_samples_per_class,
        normalize_clips=normalize_samples,
        random_state_seed=seed,
    )

    # define model and customize
    # model = architectures.DenseNet(layers_per_block=[4, 8, 8, 4], growth_rate=12)
    model = architectures.DenseNet(layers_per_block=[4, 4, 4], preproc=[("Conv2D", {"filters": 16})], dense_layers=[32])

    # perform training
    history = train(
        data_feeder=data_feeder,
        model_dir=model_dir,
        data_settings={"audio_settings": audio_settings, "spec_settings": spectral_settings},
        model_architecture=model,
        training_config=training_settings,
        verbose=1,
        random_seed=seed,
    )

    # plot learning curves
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_epochs"], history["loss"], history["eval_epochs"], history["val_loss"])
    plt.legend(["loss", "val_loss"])
    plt.ylim([0, max(plt.ylim())])
    plt.xlabel("Epoch")
    plt.ylabel(f"Loss [{'Weighted ' if training_settings['weighted_loss'] else ''}CrossEntropy]")

    plt.subplot(1, 2, 2)
    plt.plot(
        history["train_epochs"],
        history["binary_accuracy"],
        history["eval_epochs"],
        history["val_binary_accuracy"],
    )
    plt.legend(["accuracy", "val_accuracy"])
    plt.ylim([0, 1])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(Path(model_dir) / Path("learning_curve.png"), dpi=150)
    plt.close()

    # TODO: copy settings to model_dir
