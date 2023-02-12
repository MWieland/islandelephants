import matplotlib.pyplot as plt

from koogu.data import feeder
from koogu.model import architectures
from koogu import train
from pathlib import Path


def run(
    train_dir,
    validation_split,
    max_samples_per_class,
    audio_settings,
    spectral_settings,
    training_settings,
    model_dir,
    seed,
):
    # define data feeder that transforms (waveform to spectrogram) and loads prepared audio samples
    data_feeder = feeder.SpectralDataFeeder(
        data_dir=train_dir,
        fs=audio_settings["desired_fs"],
        spec_settings=spectral_settings,
        validation_split=validation_split,
        max_clips_per_class=max_samples_per_class,
    )

    # define model and customize
    model = architectures.DenseNet(layers_per_block=[4, 8, 8, 4], growth_rate=12)

    # combine audio and spectrogram settings for convenience
    data_settings = {"audio_settings": audio_settings, "spec_settings": spectral_settings}

    # perform training
    history = train(
        data_feeder=data_feeder,
        model_dir=model_dir,
        data_settings=data_settings,
        model_architecture=model,
        training_config=training_settings,
        verbose=1,
        random_seed=seed,
    )

    # plot learning curves
    # TODO: add grid and legend
    fig, ax = plt.subplots(2, sharex=True, figsize=(12, 9))
    ax[0].plot(
        history["train_epochs"],
        history["binary_accuracy"],
        "red",
        history["eval_epochs"],
        history["val_binary_accuracy"],
        "green",
    )
    ax[0].set_ylabel("Accuracy")
    ax[0].set_title("Learning Curve")
    ax[1].plot(history["train_epochs"], history["loss"], "red", history["eval_epochs"], history["val_loss"], "green")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    plt.savefig(Path(model_dir) / Path("learning_curve.png"), dpi=150)
    plt.close()
