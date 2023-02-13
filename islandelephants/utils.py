import csv
import logging
import os
import numpy as np
import random
import tensorflow as tf
from pathlib import Path


def set_deterministic(seed):
    # set deterministic behaviour for reproducible experiments
    # TODO: check that this covers all random aspects
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    np.random.seed(seed)
    tf.random.set_seed(seed)


def audio_to_annotations(audio_to_annotations_csv, audio_dir, annotations_dir):
    # map audio files to annotation files
    with open(audio_to_annotations_csv, newline="") as f:
        audio_to_annotations_list_ = list(csv.reader(f))[1:]
    audio_to_annotations_list = []
    for l in audio_to_annotations_list_:
        if Path(l[0]).suffix == ".txt":
            # put audio first and annotation last
            l[0], l[1] = l[1], l[0]
        if Path(Path(audio_dir) / Path(l[0])).is_file() and Path(Path(annotations_dir) / Path(l[1])).is_file():
            # only add sample if audio and annotations files exist
            audio_to_annotations_list.append([l[0], l[1]])
        else:
            logging.info(f"Skipping sample {l}. Audio or annotations file does not exist.")
    return audio_to_annotations_list
