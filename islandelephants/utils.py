import csv
import logging
import random

from pathlib import Path


def set_deterministic(seed):
    # TODO: expand this method to fix all randomness
    random.seed(seed)


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
