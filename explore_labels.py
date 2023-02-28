import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path


def plot_column1_per_column2(dataframe, column1, column2, plot_name, figsize, logscale=True, cnt_obs=False):
    # plots distribution and summary statistics of clip_length per sound_category
    boxplot = df.boxplot(column=column1, by=column2, figsize=figsize)
    plt.suptitle(f"{column1} per {column2}")
    boxplot.set_ylabel(f"{column1}")
    boxplot.set_xlabel("")
    boxplot.set_axisbelow(True)
    if logscale:
        boxplot.set_yscale("log")
    boxplot.yaxis.grid(color="gray", linestyle="dashed")
    boxplot.xaxis.grid(False)
    if cnt_obs:
        # calculate and plot number of observations per group
        df2 = dataframe.groupby([column2])[column1].count()
        nobs = [f"n: {str(x)}" for x in df2.tolist()]
        labels = []
        for i, label in enumerate(boxplot.get_xticklabels()):
            labels.append(f"{label.get_text()} \n ({nobs[i]})")
        boxplot.set_xticklabels(labels)
    plt.title("")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / Path(f"{plot_name}.png"), dpi=300)
    plt.close()
    print(
        f"{column1} (overall): "
        f"{round(dataframe[column1].mean(), 2)} (mean) | "
        f"{round(dataframe[column1].std(), 2)} (std) | "
        f"{round(dataframe[column1].median(), 2)} (median) | "
        f"{round(dataframe[column1].min(), 2)} (min) | "
        f"{round(dataframe[column1].max(), 2)} (max)"
    )
    dataframe = None
    df2 = None


# user settings #######################################################################################
annotation_csv = "/media/datadrive/Datasets/referencedata/islandelephants/tangkahan_original/labels.csv"
out_dir = "/media/datadrive/Datasets/referencedata/islandelephants/summary"
######################################################################################################


# load annotations from csv file
df = pd.read_csv(annotation_csv)

# how many annotated samples and annotations files do we have?
n_annotations_files = len(df.groupby(["selection_table"]))
n_samples = len(df)
print(f"Found {n_samples} annotated samples across {n_annotations_files} annotation file(s)")

# how long are annotated clips per sound_type
# how many samples do we have per category?
df["clip_length"] = df["End Time (s)"] - df["Begin Time (s)"]
plot_column1_per_column2(
    df,
    column1="clip_length",
    column2="sound_type",
    plot_name="length_type_raw",
    figsize=(15, 5),
    logscale=True,
    cnt_obs=True,
)

# which annotation files contain clips that are exceptionally long (>60s)?
limit = 60
df2 = df[df["clip_length"] > limit].groupby(["selection_table"])["clip_length"].count()
n_annotations_files_limit = len(df2)
print(f"Found {n_annotations_files_limit} annotation file(s) with clips that exceed a clip_length of {60}s")
print(df2)

# remove identified annotation files that contain exceptionally long clips (>60s)
df = df[df["clip_length"] < limit]
print(f"Removed {n_annotations_files_limit} annotation file(s) that cover(s) {n_samples - len(df)} samples")

# how long are annotated clips per sound_type after removing outliers?
# how many samples do we have per sound_type after removing outliers?
plot_column1_per_column2(
    df,
    column1="clip_length",
    column2="sound_type",
    plot_name="length_type_filtered",
    figsize=(15, 5),
    logscale=True,
    cnt_obs=True,
)

# how long are annotated clips per sound_category?
plot_column1_per_column2(
    df,
    column1="clip_length",
    column2="sound_category",
    plot_name="length_category_filtered",
    figsize=(5, 5),
    logscale=True,
    cnt_obs=True,
)

# explore annotations bandwidth (min_freq and max_freq) per sound_type and sound_category
plot_column1_per_column2(
    df,
    column1="Low Freq (Hz)",
    column2="sound_type",
    plot_name="lowfreq_type_filtered",
    figsize=(15, 5),
    logscale=True,
    cnt_obs=True,
)

plot_column1_per_column2(
    df,
    column1="Low Freq (Hz)",
    column2="sound_category",
    plot_name="lowfreq_category_filtered",
    figsize=(5, 5),
    logscale=True,
    cnt_obs=True,
)

plot_column1_per_column2(
    df,
    column1="High Freq (Hz)",
    column2="sound_type",
    plot_name="highfreq_type_filtered",
    figsize=(15, 5),
    logscale=True,
    cnt_obs=True,
)

plot_column1_per_column2(
    df,
    column1="High Freq (Hz)",
    column2="sound_category",
    plot_name="highfreq_category_filtered",
    figsize=(5, 5),
    logscale=True,
    cnt_obs=True,
)
