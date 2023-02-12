import matplotlib.pyplot as plt

from koogu import recognize, assessments

from .utils import audio_to_annotations


def run(
    audio_dir,
    annotations_dir,
    audio_to_annotations_csv,
    model_dir,
    out_dir,
    batch_size,
    label_column_name,
    negative_class_label,
):
    # predict on audio_dir and save the class probabilities
    recognize(
        model_dir=model_dir,
        audio_root=audio_dir,
        output_dir=None,
        raw_detections_dir=out_dir,
        batch_size=batch_size,
        recursive=True,
        show_progress=True,
    )

    # map audio files to annotation files
    audio_to_annotations_list = audio_to_annotations(
        audio_to_annotations_csv=audio_to_annotations_csv, audio_dir=audio_dir, annotations_dir=annotations_dir
    )

    # initialize a metric
    metric = assessments.PrecisionRecall(
        audio_annot_list=audio_to_annotations_list,
        raw_results_root=out_dir,
        annots_root=annotations_dir,
        label_column_name=label_column_name,
        negative_class_label=negative_class_label,
    )

    # run the accuracy assessments
    per_class_pr, overall_pr = metric.assess()

    # plot per class precision recall curves
    for class_name, pr in per_class_pr.items():
        plt.plot(pr["recall"], pr["precision"], "rd-")
        plt.title(f"Precision Recall Curve ({class_name}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid()
        plt.show()

    # TODO: save plot to file and add auc metric with best threshold
    # TODO: save metric results to file
