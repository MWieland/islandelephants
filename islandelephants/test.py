import matplotlib.pyplot as plt

from koogu import recognize, assessments
from pathlib import Path

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

    # plot per precision recall curves (per class and overall)
    # TODO: add auc metric with best threshold
    colors = ["red", "green"]
    for i, item in enumerate(per_class_pr.items()):
        class_name, pr = item
        plt.plot(pr["recall"], pr["precision"], color=colors[i], label=class_name)
    plt.plot(overall_pr["recall"], overall_pr["precision"], color="blue", label="overall")
    plt.title(f"Precision Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid()
    plt.savefig(Path(out_dir) / Path("precision_recall_curve.png"), dpi=150)
    plt.close()
