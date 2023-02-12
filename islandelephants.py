import argparse
import datetime
import logging
import yaml

from pathlib import Path

from islandelephants import predict, prepare, test, train, utils


parser = argparse.ArgumentParser(
    description="""
.-. .----..-.     .--.  .-. .-..----.    .----..-.   .----..----. .-. .-.  .--.  .-. .-. .---.  .----.
| |{ {__  | |    / {} \ |  `| || {}  \   | {_  | |   | {_  | {}  }| {_} | / {} \ |  `| |{_   _}{ {__  
| |.-._} }| `--./  /\  \| |\  ||     /   | {__ | `--.| {__ | .--' | { } |/  /\  \| |\  |  | |  .-._} }
`-'`----' `----'`-'  `-'`-' `-'`----'    `----'`----'`----'`-'    `-' `-'`-'  `-'`-' `-'  `-'  `----' 
islandelephants provides routines for audio classification. Modules for data preparation, training, 
testing and prediction are provided. Focus is on working with audio recordings of elephant sounds. 
""",
    formatter_class=argparse.RawTextHelpFormatter,
)

parser.add_argument(
    "--settings",
    help=f"Path to YAML file with settings",
    required=True,
)

parser.add_argument(
    "--prepare",
    action="store_true",
    help=f"Prepare data for training an audio classification model",
    required=False,
)

parser.add_argument(
    "--train",
    action="store_true",
    help=f"Train model for audio classification",
    required=False,
)

parser.add_argument(
    "--predict",
    action="store_true",
    help=f"Run inference on directory of WAV audio files",
    required=False,
)

parser.add_argument(
    "--test",
    action="store_true",
    help=f"Test performance of predictions",
    required=False,
)

args = parser.parse_args()

if Path(args.settings).is_file():
    with open(args.settings, "r") as f:
        settings = yaml.safe_load(f)
else:
    raise Exception("Cannot find settings YAML file.")

logging.basicConfig(
    filename=settings["GENERAL"]["LOGFILE"] if True else None,
    filemode="w",
    format="%(levelname)s: %(message)s",
    level=settings["GENERAL"]["LOGLEVEL"],
)

utils.set_deterministic(settings["GENERAL"]["SEED"])

starttime = datetime.datetime.now()
logging.info("Starttime: " + str(starttime.strftime("%H:%M:%S")))

if args.prepare:
    prepare.run(
        audio_dir=settings["PREPARE"]["AUDIO_DIR"],
        annotations_dir=settings["PREPARE"]["ANNOTATIONS_DIR"],
        audio_to_annotations_csv=settings["PREPARE"]["AUDIO_TO_ANNOTATIONS_CSV"],
        out_dir=settings["PREPARE"]["OUT_DIR"],
        label_column_name=settings["PREPARE"]["LABEL_COLUMN_NAME"],
        negative_class_label=settings["PREPARE"]["NEGATIVE_CLASS_LABEL"],
        audio_settings=settings["PREPARE"]["AUDIO_SETTINGS"],
        train_test_split=settings["PREPARE"]["TRAIN_TEST_SPLIT"],
    )

if args.train:
    train.run(
        train_dir=settings["TRAIN"]["TRAIN_DIR"],
        validation_split=settings["TRAIN"]["VALIDATION_SPLIT"],
        max_samples_per_class=settings["TRAIN"]["MAX_SAMPLES_PER_CLASS"],
        audio_settings=settings["PREPARE"]["AUDIO_SETTINGS"],
        spectral_settings=settings["PREPARE"]["SPECTRAL_SETTINGS"],
        training_settings=settings["TRAIN"]["TRAINING_SETTINGS"],
        model_dir=settings["TRAIN"]["MODEL_DIR"],
        seed=settings["GENERAL"]["SEED"],
    )

if args.test:
    test.run(
        audio_dir=settings["TEST"]["AUDIO_DIR"],
        annotations_dir=settings["TEST"]["ANNOTATIONS_DIR"],
        audio_to_annotations_csv=settings["TEST"]["AUDIO_TO_ANNOTATIONS_CSV"],
        model_dir=settings["TEST"]["MODEL_DIR"],
        out_dir=settings["TEST"]["OUT_DIR"],
        batch_size=settings["TEST"]["BATCH_SIZE"],
        label_column_name=settings["TEST"]["LABEL_COLUMN_NAME"],
        negative_class_label=settings["TEST"]["NEGATIVE_CLASS_LABEL"],
    )

if args.predict:
    predict.run(
        predict_dir=settings["PREDICT"]["PREDICT_DIR"],
    )

endtime = datetime.datetime.now()
logging.info("Endtime : " + str(endtime.strftime("%H:%M:%S")))
logging.info(str((endtime - starttime).total_seconds()) + " sec")
