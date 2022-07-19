# GoMatch: A Geometric-Only Matcher for Visual Localization

Authors: Qunjie Zhou [@GrumpyZhou](https://github.com/GrumpyZhou), Sérgio Agostinho [@SergioRAgostinho](https://github.com/SergioRAgostinho)

## ❗️Important Comment❗️
Due to the time constraints of ECCV camera ready deadline, we are still working on preparing the pretrained models and more detailed instructions about reproducing our experiments. Please be patient about the update and we will try to do that asap. Thanks for your understanding. 

## Setting Up Special Dependencies

Despite `pip` handling most of the dependency installation process there are some special packages that **usually** should be installed in advance, e.g. `pytorch` and `torch-scatter`. We list them below. Please follow their installation instructions first before proceeding.

Special packages:
- pytorch:
    - installation: https://pytorch.org/get-started/locally/
    - version: `1.7.1  py3.7_cuda10.2.89_cudnn7.6.5_0` (note the cuda and cudnn versions)
- pytorch-scatter:
    - installation: https://github.com/rusty1s/pytorch_scatter#installation
    - version: `2.0.6`

## Installing GoMatch

To install GoMatch, simply navigate to its root folder in your terminal and type
```
pip install .
```

This will pull all dependencies and install GoMatch's package with its models so that you can easily use them in your project.

## Uninstalling GoMatch

Type in your terminal
```
pip uninstall gomatch
```
