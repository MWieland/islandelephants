import csv
import logging
import random

from koogu.data import preprocess
from pathlib import Path


def run(
    audio_dir,
    annotations_dir,
    audio_to_annotations_csv,
    out_dir,
    negative_class_label,
    audio_settings,
    train_test_split,
):
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
            audio_to_annotations_list.append([Path(l[0]).stem, l[1]])
        else:
            logging.info(f"Skipping sample {l}. Audio or annotations file does not exist.")

    if train_test_split:
        # split audio_to_annotations_list into train and test
        random.shuffle(audio_to_annotations_list)
        train_audio_to_annotations_list = audio_to_annotations_list[
            0 : int(len(audio_to_annotations_list) * train_test_split[0])
        ]
        test_audio_to_annotations_list = audio_to_annotations_list[
            int(len(audio_to_annotations_list) * train_test_split[0]) :
        ]
        out_dirs = [str(Path(out_dir) / Path("train")), str(Path(out_dir) / Path("test"))]
        audio_to_annotations_lists = [train_audio_to_annotations_list, test_audio_to_annotations_list]
    else:
        # leave audio_to_annotations_list as is
        out_dirs = [out_dir]
        audio_to_annotations_lists = [audio_to_annotations_list]

    # do the actual prprocessing
    for i in range(len(out_dirs)):
        clip_counts = preprocess.from_selection_table_map(
            audio_settings=audio_settings,
            audio_seltab_list=audio_to_annotations_lists[i],
            audio_dir=audio_dir,
            seltab_root=annotations_dir,
            output_root=out_dirs[i],
            negative_class_label=negative_class_label,
        )
        [logging.info(f"{label:<10s}: {count:d}") for label, count in clip_counts.items()]
