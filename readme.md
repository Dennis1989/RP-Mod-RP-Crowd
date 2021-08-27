# RP-Mod & RP-Crowd: Moderator- and Crowd-Annotated German News Comment Datasets

This repository contains all evaluation scripts of the conducted experiments in our recent work that can be accessed via [OpenReview.net]( https://openreview.net/forum?id=NfTU-wN8Uo).
All evaluation scripts are stored in dedicated `Jupyter` Notebooks.

## Preparation

Before the scripts can be executed, the corresponding RP datasets have to be included in the subfolder:

[`Dataset/Text-Data/`](Dataset/Text-Data/)


Here is an example of the final contents of the folder:

```asciidoc
├── Dataset
    ├── Text-Data
        ├── RP_Mod-Crowd.csv
        ├── RP-Mod.csv
        ├── RP-Mod-folds.csv
        ├── RP-Crowd-5.csv
        ├── RP-Crowd-5-folds.
        ├── RP-Crowd-4.csv
        ├── RP-Crowd-4-folds.csv
        ├── RP-Crowd-3.csv
        ├── RP-Crowd-3-folds.csv
        ├── RP-Crowd-2.csv
        ├── RP-Crowd-2-folds.csv
        ├── RP-Crowd-1.csv
        ├── RP-Crowd-1-folds.csv
        ├── derstandard.csv
        ├── derstandard-folds.csv
        ├── CrowdGuru-Demographic.xlsx
        ├── CrowdGuru-Ratings.xlsx
├── Evaluation
├── Preprocessing
├── Training

```

Our datasets are hosted on [zenodo](https://zenodo.org/) and can be retrieved here:

[https://zenodo.org/record/5242915](https://zenodo.org/record/5242915)

Currently, access is granted on an *individual basis* and has to be requested. We will make the dataset accessible to the public in the future.

## Structure

The repository consists of the following sections:

1. Dataset 
2. Preprocessing
3. Training
4. Evaluation

The [`Dataset`](Dataset/) subfolder contains all the raw and processed data used for descriptive analyses and model training (explanations regarding the origin and preprocessing of the `DerStandard` data in [`derstandard.csv`](Dataset/Text-Data/derstandard.csv) and [`derstandard-folds.csv`](Dataset/Text-Data/derstandard-folds.csv) can be found in the corresponding [README](Dataset/Text-Data/README.md)). In [`Preprocessing`](Preprocessing/), we store scripts that transform the original dataset to create subsets such as [`RP-Mod`](Dataset/RP-Mod.csv), [`RP-Crowd-1`](Dataset/RP-Crowd-1.csv), ..., [`RP-Crowd-5`](Dataset/RP-Crowd-5.csv). Also, we include scripts to create training/validation folds. The [`Training`](Training/) section itself is subdivided into *Baseline* and *Language Models*. Here we store all scripts to train our models. Last, in the [`Evaluation`](Evaluation/) section, we keep evaluation scripts. All the Figures from our work are produced here. Also, the training output is stored in this section.

## Preparation

Please make sure to create a virtual environment:
```bash
pip3 install virtualenv
virtualenv venv
source venv/bin/activate
```

Install necessary dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Since we are using autosklearn for creating AutoML pipelines, it should be manually installed according to (https://automl.github.io/auto-sklearn/master/installation.html):

```bash
curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip3 install

pip3 install auto-sklearn
```

As described in the paper, we used three Nvidia Quadro RTX 6000 with 24GB of memory. Parts of the code expects the user to have three GPUs installed. To run the code with `CUDA` support, please ensure that latest Nvidia Drivers are installed. In addition, `CUDA 11.1` and `cuDNN 8.0` must be installed:

* [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
* [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)

Afterwards, one can install `PyTorch LTS (1.8.2)` + `CUDA` which is the version that we used throughout the training:
```bash
pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```
## Important Note
Currently, there is a versioning incompatibility between `scikit-optimize` and the latest `scikit-learn` version. This incompatibility is already fixed in the current development branch. Therefore we ask to install the package from the source:

The development version can be installed through:

```bash
git clone https://github.com/scikit-optimize/scikit-optimize.git
cd scikit-optimize
pip install -e .
```

Please also make sure to download the following german corpus for preprocessing:
```
python3 -m spacy download de_core_news_lg
```
