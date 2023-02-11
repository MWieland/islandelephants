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
Koogu uses Tensorflow. To use it with GPU support check [here](https://www.tensorflow.org/install/source#tested_build_configurations) for Python and Tensorflow versions that match your CUDA and CUDNN versions. 

To check your CUDA version you can run the following.
```shell
$ whereis cuda
$ cat /usr/lib/cuda/version.txt
```


NOTE: This requires tensorflow>=2.4.0 and therefore cuda>=11 and cudnn>=8 -> may have to upgrade my cuda!!!


Create conda environment and install dependencies with version numbers matching your CUDA and CUDNN setup.
```shell
$ conda create -n islandelephants python=3.8
$ conda install -c conda-forge cudatoolkit=10.1 cudnn=7.6
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
$ pip install tensorflow==2.3.0 koogu pyaml matplotlib
```
