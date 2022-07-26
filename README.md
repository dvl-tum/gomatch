# GoMatch: A Geometric-Only Matcher for Visual Localization

Authors: Qunjie Zhou [@GrumpyZhou](https://github.com/GrumpyZhou), Sérgio Agostinho [@SergioRAgostinho](https://github.com/SergioRAgostinho)

## ❗️Important Comment❗️
Due to the time constraints of ECCV camera ready deadline, we are still working on preparing the pretrained models and more detailed instructions about reproducing our experiments. Please be patient about the update and we will try to do that asap. Thanks for your understanding. 

## Table of Contents

- [Setting Up Special Dependencies](#setting-up-special-dependencies)
- [Installing GoMatch](#installing-gomatch)
- [Uninstalling GoMatch](#uninstalling-gomatch)


## Setting Up Special Dependencies

Despite `pip` handling most of the dependency installation process there are some special packages that **usually** should be installed in advance, e.g. `pytorch` and `torch-scatter`. We list them below. Please follow their installation instructions first before proceeding.

Special packages:
- pytorch:
    - installation: https://pytorch.org/get-started/locally/
    - version: `1.7.1  py3.7_cuda10.2.89_cudnn7.6.5_0` (note the cuda and cudnn versions)
- pytorch-scatter:
    - installation: https://github.com/rusty1s/pytorch_scatter#installation
    - version: `2.0.6`

While we used these versions throughout development, one might be able to use their most recent versions and still replicate our results. We haven't tested it yet, so proceed at your own risk. If you're struggling to replicate results, consider creating an environment replicating the versions we used, as a sanity check.

## Installing GoMatch

The codebase is organized in a way that all code we considered that could be useful to other projects is bundled inside the `gomatch` top level directory. Whenever we mention installation, we are referring to the installation of this folder alongside the other packages in your python environment. The package includes the network architectures we tested, losses, metrics and the data loaders we used. It does NOT include the training and evaluation script. Those are not installed and need to be executed from the root folder of this project.

To install GoMatch, simply navigate to its root folder in your terminal and type
```
pip install .
```

This will pull all dependencies and install GoMatch's package with its models so that you can easily use them in your project.

To install all required dependencies to also run training scripts and other tools, invoke
```
pip install ".[full]"
```


## Uninstalling GoMatch

Type in your terminal
```
pip uninstall gomatch
```
If you see the following error message
> Can't uninstall 'gomatch'. No files were found to uninstall.

Just call the command outside the root folder of the project.

