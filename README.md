# islandelephants
[islandelephants]() provides modules for data preparation, training, testing and prediction of animal sounds in audio recordings. Focus is on working with recordings of elephant sounds.

> NOTE: This is work in progress and not ready to be used yet.

## Usage
Simple command line tool. To get help call the following.

```shell
$ python islandelephants.py --help
```

### Prepare dataset
This prepares a dataset for training an audio classifier to detect animal (here elephant) sounds in audio files. We currently support audio files in [Wav]() format and annotations in [Raven]() format. The following steps are covered by dataset preparation.

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


## Installation

### Docker
To avoid any issues with dependencies regarding Tensorflow, CUDA and cudnn we recommend running [islandelephants]() with [Docker](https://docs.docker.com/config/containers/resource_constraints/#gpu). Build a Docker image by running the following command.

```shell
$ docker build -f islandelephants.Dockerfile --tag islandelephants --network=host .
```

Once installed you can execute the Docker image as follows (example for training). Make sure to mount a local directory as volume. The directory should contain all required data (e.g. training audio and annotation files, settings_docker.yaml).

```shell
$ docker run --gpus '"device=0"' --shm-size 30G -d --name islandelephants_train --network=host --rm -v /my/local/data/dir/:/scratch/ islandelephants --train --settings /scratch/settings_docker.yaml
```

### Conda
Alternative to running [islandelephants]() with Docker is to create a virtual environment with [Conda](https://docs.conda.io/en/latest/miniconda.html). To use Tensorflow with GPU support check [here](https://www.tensorflow.org/install/source#tested_build_configurations) for Python and Tensorflow versions that match your CUDA and CUDNN versions. Minimum requirements are tensorflow>=2.4.0 and therefore cuda>=11 with cudnn>=8.

Example installation with conda environment
```shell
$ conda create -n islandelephants python=3.10
$ conda activate islandelephants
$ conda install -c conda-forge cudatoolkit cudnn
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
$ pip install tensorflow koogu pyaml matplotlib
```
