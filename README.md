# islandelephants
Package description.

## Dataset - Indonesian elephant sounds
Source: https://drive.google.com/drive/folders/1ofY7hXVFbogLQTukxNy5FQx0qL1oUWXs

### Prepare dataset
- Ignore samples for which no annotation or audio file could be found
- Randomly split dataset into training and testing (80 / 20) with fixed seed
- Clip and label audio files according to annotations
- Save samples as compressed numpy arrays with labels and waveforms

## TODOs
- Test simple Koogu pipeline from preparation to training and testing

## Installation
```shell
$ conda create -n islandelephants python=3.9
$ conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
$ pip install tensorflow koogu pyaml
```
