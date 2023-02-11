import matplotlib.pyplot as plt

from koogu import recognize, assessments


def run():
    # The root directories under which the test data (audio files and
    # corresponding annotation files) are available.
    test_audio_root = "/home/shyam/projects/NARW/data/test_audio"
    test_annots_root = "/home/shyam/projects/NARW/data/test_annotations"
    model_dir = None

    # Map audio files to corresponding annotation files
    test_audio_annot_list = [
        ["NOPP6_EST_20090401", "NOPP6_20090401_RW_upcalls.selections.txt"],
        ["NOPP6_EST_20090402", "NOPP6_20090402_RW_upcalls.selections.txt"],
        ["NOPP6_EST_20090403", "NOPP6_20090403_RW_upcalls.selections.txt"],
    ]

    # Directory in which raw detection scores will be saved
    raw_detections_root = "/home/shyam/projects/NARW/test_audio_raw_detections"

    # run the model and save the class probabilities
    recognize(
        model_dir,
        test_audio_root,
        raw_detections_dir=raw_detections_root,
        batch_size=64,
        recursive=True,
        show_progress=True,
    )

    # Initialize a metric object with the above info
    metric = assessments.PrecisionRecall(test_audio_annot_list, raw_detections_root, test_annots_root)

    # The metric supports several options (including setting explicit thresholds).
    # Refer to class documentation for more details.
    # Run the assessments and gather results
    per_class_pr, overall_pr = metric.assess()

    # Plot PR curves.
    for class_name, pr in per_class_pr.items():
        plt.plot(pr["recall"], pr["precision"], "rd-")
        plt.title(f"Precision Recall Curve ({class_name}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid()
        plt.show()

    # Similarly, you could plot the contents of 'overall_pr' too
