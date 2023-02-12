import csv
import logging
import random

from koogu.data import preprocess
from pathlib import Path

from .utils import audio_to_annotations


def run(
    audio_dir,
    annotations_dir,
    audio_to_annotations_csv,
    out_dir,
    label_column_name,
    negative_class_label,
    audio_settings,
    train_test_split,
):
    # map audio files to annotation files
    audio_to_annotations_list = audio_to_annotations(
        audio_to_annotations_csv=audio_to_annotations_csv, audio_dir=audio_dir, annotations_dir=annotations_dir
    )

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

    # TODO: remove this block - only temporary to test with few samples
    # audio_to_annotations_lists[0] = audio_to_annotations_lists[0][:3]
    # audio_to_annotations_lists[1] = audio_to_annotations_lists[1][:2]

    for i in range(len(out_dirs)):
        # save prepared audio_to_annotations_lists
        with open(Path(out_dirs[i]) / Path("audio_to_annotations.csv"), "w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["audio", "annotations"])
            csv_writer.writerows(audio_to_annotations_lists[i])

        # do the actual preprocessing
        clip_counts = preprocess.from_selection_table_map(
            audio_settings=audio_settings,
            audio_seltab_list=audio_to_annotations_lists[i],
            audio_root=audio_dir,
            seltab_root=annotations_dir,
            output_root=out_dirs[i],
            negative_class_label=negative_class_label,
            label_column_name=label_column_name,
        )
        [logging.info(f"{label:<10s}: {count:d}") for label, count in clip_counts.items()]
