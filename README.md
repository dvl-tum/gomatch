# GoMatch: A Geometric-Only Matcher for Visual Localization

Authors: Qunjie Zhou [@GrumpyZhou](https://github.com/GrumpyZhou), Sérgio Agostinho [@SergioRAgostinho](https://github.com/SergioRAgostinho)

This repository contains the Pytorch implementation of our paper accepted at ECCV22: Is Geometry Enough for Matching in Visual Localization? [[Paper]](https://arxiv.org/pdf/2203.12979.pdf) [[Video]](https://drive.google.com/file/d/1Rj_5PdIBsCVLNGrefEojTq6FcHLVEFqt/view?usp=sharing) [[Poster]](https://drive.google.com/file/d/1q_817QudISWF-LR5MtA9aL-vddUuGHpu/view?usp=sharing).


## Table of Contents

- [Installing GoMatch](#installing-gomatch)
- [Uninstalling GoMatch](#uninstalling-gomatch)

## Installing GoMatch

We rely on conda and pip for installing our package.

TL;DR:
```bash
conda create -n gomatch python=3.7
conda install pytorch==1.7.1 cudatoolkit=10.2 -c pytorch
pip install . --find-links https://data.pyg.org/whl/torch-1.7.1+cu102.html
```

The codebase is organized in a way that all code we considered that could be useful to other projects is bundled inside the `gomatch` top level directory. Whenever we mention installation, we are referring to the installation of this folder alongside the other packages in your python environment. The package includes the network architectures we tested, losses, metrics and the data loaders we used. It does NOT include the training and evaluation script. Those are not installed and need to be executed from the root folder of this project.

Once PyTorch is installed through conda, to install GoMatch, simply navigate to its root folder in your terminal and type
```
pip install . --find-links https://data.pyg.org/whl/torch-1.7.1+cu102.html
```

This will pull all dependencies and install GoMatch's package with its models so that you can easily use them in your project.

To install all required dependencies to also run training scripts and other tools, invoke
```
pip install ".[full]" --find-links https://data.pyg.org/whl/torch-1.7.1+cu102.html
```


## Uninstalling GoMatch

Type in your terminal
```
pip uninstall gomatch
```
If you see the following error message
> Can't uninstall 'gomatch'. No files were found to uninstall.

Just call the command outside the root folder of the project.

## Evaluation

#### Pretained model
The pretrained models can be manually downloaded from [here](https://drive.google.com/file/d/1-J4SEBL6tBu3OpSSpW7qi_VA6LVcK98q/view?usp=sharing).
Or download it from command line using gdown:
```
gdown 1-J4SEBL6tBu3OpSSpW7qi_VA6LVcK98q
```
The zip file contains the following three models:
- _BPnPNet.ckpt_: our re-trained BPnPNet in our environment.
- _GoMatchBVs_best.ckpt_: our best GoMatch model using bearing vectors.
- _GoMatchCoords_best.ckpt_: our best GoMatch architecture trained using 3D coordinates.
