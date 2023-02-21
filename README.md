# GoMatch: A Geometric-Only Matcher for Visual Localization

Authors: Qunjie Zhou* [@GrumpyZhou](https://github.com/GrumpyZhou), Sérgio Agostinho* [@SergioRAgostinho](https://github.com/SergioRAgostinho), Aljosa Osep and Laura Leal-Taixé. (* equal contribution)

This repository contains the Pytorch implementation of our paper accepted at ECCV22: Is Geometry Enough for Matching in Visual Localization? [[Paper]](https://arxiv.org/pdf/2203.12979.pdf) [[Video]](https://youtu.be/wgAOJlb2uD4) [[Poster]](https://drive.google.com/file/d/1q_817QudISWF-LR5MtA9aL-vddUuGHpu/view?usp=sharing).


## Table of Contents

- [Installing GoMatch](#installing-gomatch)
- [Uninstalling GoMatch](#uninstalling-gomatch)
- [Data Preparation](#data-preparation)
- [Evaluation](#evaluation)
- [Training](#training)

## Installing GoMatch

We rely on conda and pip for installing our package.

TL;DR:
```bash
conda create -n gomatch python=3.7
conda install pytorch==1.7.1 cudatoolkit=10.2 -c pytorch
pip install . --find-links https://data.pyg.org/whl/torch-1.7.1+cu102.html
```
We also exported our conda environment in [conda_env_reproduce.yml](conda_env_reproduce.yml).

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


## Data Preparation
Data preparation steps for training and evaluation are described in [tools/README.md](tools/README.md). Make sure you check this out and follow the instructions before you tryout the training and evaluation.

## Evaluation

#### Pretained model
The pretrained models can be manually downloaded from [this url](https://vision.in.tum.de/webshare/u/zhouq/gomatch/pretrained) or [google drive](https://drive.google.com/file/d/1-J4SEBL6tBu3OpSSpW7qi_VA6LVcK98q/view?usp=sharing).
To download from Gdrive from command line:
```
# Install gdown first
gdown 1-J4SEBL6tBu3OpSSpW7qi_VA6LVcK98q
```


The zip file contains the following three models:
- _BPnPNet.ckpt_: our re-trained BPnPNet in our environment.
- _GoMatchBVs_best.ckpt_: our best GoMatch model using bearing vectors.
- _GoMatchCoords_best.ckpt_: our best GoMatch architecture trained using 3D coordinates.

*Comments*:
In our code examples, we stored the pretrained models under _outputs/shared_outputs/release_models/exported_models_. Make sure you update the directory accordingly before you execute them.


#### Example notebook
To evalute a pretrained model on various localization benchmarks, we present code examples in [notebooks/eval_examples.ipynb](notebooks/eval_examples.ipynb). 

You can also exeucte evaluation from command line, use the following commands. We use [configs/datasets.yml](configs/datasets.yml) to set dataset paths. 
```
# Define ckpt path of our best model GoMatchBVs
gomatchbvs='outputs/shared_outputs/release_models/exported_models/GoMatchBVs_best.ckpt'

# Eval On Megadepth 
python -m gomatch_eval.benchmark  --root .  --ckpt $gomatchbvs \
    --splits 'test'  \
    --odir 'outputs/benchmark_cache_release' \
    --dataset 'megadepth' --covis_k_nums 10  \
    --p2d_type 'sift'  

# Eval On Cambridge Landmarks
python -m gomatch_eval.benchmark  --root .  --ckpt $gomatchbvs \
    --splits  'kings' 'old' 'shop' 'stmarys' \
    --odir 'outputs/benchmark_cache_release' \
    --dataset 'cambridge_sift' --covis_k_nums 10  \
    --p2d_type 'sift'

# Eval On 7 Scenes 
python -m gomatch_eval.benchmark  --root .  --ckpt $gomatchbvs \
    --splits 'chess' 'fire' 'heads' 'office' 'pumpkin' 'redkitchen' 'stairs'  \
    --odir 'outputs/benchmark_cache_release' \
    --dataset '7scenes_superpoint_v2' --covis_k_nums 10  \
    --p2d_type 'superpoint'
```

To evaluate the visual descriptor matching baselines, specify '--vismatch' as below:
```
# Eval on Cambridge Landmarks
python -m gomatch_eval.benchmark  --root .  --vismatch \
    --splits  'kings' 'old' 'shop' 'stmarys' \
    --odir 'outputs/benchmark_cache_release' \
    --dataset 'cambridge_sift' --covis_k_nums 10  \
    --p2d_type 'sift'

# Eval On 7 Scenes 
python -m gomatch_eval.benchmark  --root .  --vismatch \
    --splits 'chess' 'fire' 'heads' 'office' 'pumpkin' 'redkitchen' 'stairs'  \
    --odir 'outputs/benchmark_cache_release' \
    --dataset '7scenes_superpoint_v2' --covis_k_nums 10  \
    --p2d_type 'superpoint'
    
```    


## Training

We train GoMatch (i.e., our best model GoMatchBVs) on a single 48GB Nvidia rtx_8000 GPU with the following command:
```
python -m gomatch_train.train_matcher --gpus 0 --batch 16 -lr 0.001 \
    --max_epochs 50 --matcher_class 'OTMatcherCls' --share_kp2d_enc \
    --dataset 'megadepth' --train_split 'train' --val_split 'val' \
    --outlier_rate 0.5 0.5  --topk 1 --npts 100 1024 \
    --p2d_type 'sift' --p3d_type 'bvs' \
    --inls2d_thres 0.001 --rpthres 0.01 --prefix 'gomatchbvs' \
    -o 'outputs/eccv22' --num_workers 4
```
We train BPnPNet baseline on a single 12GB Nvidia titan x GPU with the following command:
```
python -m gomatch_train.train_matcher --gpus 0 --batch 64 -lr 0.001 \
    --max_epochs 50 --matcher_class 'BPnPMatcher' \
    --dataset 'megadepth' --train_split 'train' --val_split 'val' \
    --outlier_rate 0.0 0.0  --topk 5 --npts 100 1024 \
    --p2d_type 'sift' --p3d_type 'coords' \
    --inls2d_thres 0.001 --rpthres 0.01 --prefix 'bpnpnet' \
    -o 'outputs/eccv22' --num_workers 4
```
