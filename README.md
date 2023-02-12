# islandelephants
Package description.

> NOTE: This is work in progress and not ready to be used yet.

## Usage
Simple command line tool.

```shell
$ python islandelephants.py --help
```

### Prepare dataset
This prepares a dataset for training and audio classifier to detect animal (here elephant) sounds in audio files. We currently support audio files in [Wav]() format and annotations in [Raven]() format. The following steps are covered by dataset preparation.

- Ignore samples for which no annotation or audio file could be found
- Randomly split dataset into training and testing (80 / 20) with fixed seed
- Clip and label audio files according to annotations and extract samples
- Save samples as compressed numpy arrays with labels and waveforms

```shell
$ python islandelephants.py --prepare --settings settings.yaml
```

### Train audio classifier
- ...

```shell
$ python islandelephants.py --train --settings settings.yaml
```

### Test audio classifier
- ... 

```shell
$ python islandelephants.py --test --settings settings.yaml
```

### Detect elephant sounds in audio samples
- ...

```shell
$ python islandelephants.py --predict --settings settings.yaml
```

## Sumatran elephants dataset
Source: https://drive.google.com/drive/folders/1ofY7hXVFbogLQTukxNy5FQx0qL1oUWXs

## Installation
To use [islandelephants]() with GPU support check [here](https://www.tensorflow.org/install/source#tested_build_configurations) for Python and Tensorflow versions that match your CUDA and CUDNN versions. Minimum requirements are tensorflow>=2.4.0 and therefore cuda>=11 and cudnn>=8.

Example installation with conda environment
```shell
$ conda create -n islandelephants python=3.10
$ conda install -c conda-forge cudatoolkit cudnn
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
$ pip install tensorflow koogu pyaml matplotlib
```
