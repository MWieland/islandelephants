import pandas as pd
import shutil

from pathlib import Path


# user settings #######################################################################################
in_dir = "/mnt/s3_islandelephants"
audio_to_annotations_csv = "/mnt/s3_islandelephants/labels/selection_tables_to_soundfiles.csv"
exclude_annotations_files = ["0mCage_Eloc_12102021_005017.Table.1.selections.txt"]
out_dir = "/media/datadrive/Datasets/referencedata/islandelephants/tangkahan_original"
######################################################################################################

# create audio to annotations list and remove exclude files
df = pd.read_csv(audio_to_annotations_csv)
print(f"Found {len(df)} annotations files")

for i in range(len(exclude_annotations_files)):
    df = df[df["selection_table"] != exclude_annotations_files[i]]
print(f"Excluded {len(exclude_annotations_files)} annotations file(s). Now we have {len(df)} annotations files")

# make output subdirectories (audio and annotations)
Path(Path(out_dir) / Path("audio")).mkdir(parents=True, exist_ok=True)
Path(Path(out_dir) / Path("annotations")).mkdir(parents=True, exist_ok=True)

df = df.reset_index()
dst_audio_files, dst_annotations_files = [], []
for index, row in df.iterrows():
    # get src and dst files and deal with csv ending on txt files
    src_audio_file = Path(in_dir) / Path(row["soundfile_directory"])
    src_annotations_file = Path(in_dir) / Path(f"{row['selection_table_directory']}.csv")
    dst_audio_file = Path(out_dir) / Path("audio") / src_audio_file.name
    dst_annotations_file = Path(out_dir) / Path("annotations") / Path(src_annotations_file.name)  # .with_suffix("")
    if Path(src_audio_file).is_file() and Path(src_annotations_file).is_file():
        # copy audio and annotations files to out_dir
        # if not dst_audio_file.is_file():
        #    shutil.copyfile(src=src_audio_file, dst=dst_audio_file)
        # if not dst_annotations_file.is_file():
        #    shutil.copyfile(src=src_annotations_file, dst=dst_annotations_file)
        dst_audio_files.append(dst_audio_file.name)
        dst_annotations_files.append(str(Path(dst_annotations_file.name).with_suffix("")))
    else:
        print(f"Could not find both {src_audio_file} and {src_annotations_file} files. Skipping them.")

# save updated audio_to_annotations_csv
df2 = pd.DataFrame(list(zip(dst_audio_files, dst_annotations_files)), columns=["audio_file", "annotations_file"])
df2.to_csv(Path(Path(out_dir) / Path("audio_to_annotations.csv")), index=False)

for dst_annotations_file in Path(Path(out_dir) / Path("annotations")).glob("*.csv"):
    # format copied .txt.csv annotations files to true .txt files as required by koogu
    pd.read_csv(dst_annotations_file, index_col=0).to_csv(
        Path(out_dir) / Path("annotations") / dst_annotations_file.stem, sep="\t", index=False
    )
    # remove .txt.csv annotations file
    Path(dst_annotations_file).unlink()

"""
# Questions
1. Why selection_tables_to_soundfiles.csv has 220 files?
-> there are 216 files referenced in labels.csv (used as basis for explore_labels.py statistics)
-> there are 244 files referenced in selection_tables.csv
-> which one to use?
-> after filtering out non existing files (wav or txt) we get 216 and after filtering out badly annotated files we get 215
-> seems correct like this
2. Why selection tables are referenced as .txt but saved as .txt.csv?
"""
