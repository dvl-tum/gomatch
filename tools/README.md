# Data Preparation Steps

All data processing tools assume by default a symlink to the folder `data`, linking each dataset: MegaDepth, Cambridge Landmarks and 7 scenes. Here's an example on how that looks normally

```
$ tree data
data
├── 7scenes -> /some_path_to/7scenes
├── cambridge -> /some_path_to/cambridge
├── MegaDepth_undistort -> /some_path_to/megadepth
```

You also need to make use of a third-party dependency called `immatch`, that is available [here](https://github.com/GrumpyZhou/image-matching-toolbox) to generate descriptor caches.

## MegaDepth

### Getting the Dataset

Download and preprocess the dataset following the [instructions](https://github.com/mihaidusmanu/d2-net#downloading-and-preprocessing-the-megadepth-dataset) from the D2-Net project repo.

### Generating a Scene File

tool location: [`tools/prep_megadepth_2D3D_data.py`](tools/prep_megadepth_2D3D_data.py)

#### Option 1
By default the tool assumes there is a folder with the undistorted SfM reconstruction of MegaDepth symlinked to the `data` folder, as explained in the previous section. If the folder is symlinked, theoretically no arguments need to passed. Just call the tool with
```
$ python tools/prep_megadepth_2D3D_data.py
```

After completion, the tool will generate a number of files in `data/MegaDepth_undistort/data_processed/v2`.

#### Option 2
You can also down load the exact file we used for training from [here](https://vision.in.tum.de/webshare/u/zhouq/gomatch/train_data/). Notice, the name of the file might differ from the one if you generate since we have cleanned some outdated code in the script. But the logic remains unchanged.
After that you still need to generate the cached 3D points using the following command:
```
python tools/prep_megadepth_2D3D_data.py --save-dir 'data/MegaDepth_undistort/data_processed/v2' \
    --skip-scene-parsing --save-pickle --split-config 'configs/datasets.yml'
```
The option `--save-pickle` saves the data as pickle file otherwise by default as a numpy file. 

### Generating a Descriptor Cache

Run the tool `tools/extract_features_immatch.py`.

```
python -m tools.extract_features_immatch --immatch_config 'immatch/sift'   --dataset 'megadepth'
```
This will generate the descriptor cache for sift.

## Cambridge Landmarks
### Getting the Dataset

The dataset is composed of multiple landmarks. These can be downloaded from the following locations:
- [Great Court](https://www.repository.cam.ac.uk/handle/1810/251291)
- [King's College](https://www.repository.cam.ac.uk/handle/1810/251342)
- [Old Hospital](https://www.repository.cam.ac.uk/handle/1810/251340)
- [Shop Facade](https://www.repository.cam.ac.uk/handle/1810/251336)
- [St. Mary's Church](https://www.repository.cam.ac.uk/handle/1810/251294)

Download the reconstructions and retrieval pairs released by [PixLoc](https://github.com/cvg/pixloc) at this [url](https://cvg-data.inf.ethz.ch/pixloc_CVPR2021/Cambridge-Landmarks/). You can download everything recursively using the following command
```
wget -r -np -R "index.html*" -nH --cut-dirs=2 https://cvg-data.inf.ethz.ch/pixloc_CVPR2021/Cambridge-Landmarks/
```
If you call this from the folder where your extracted your dataset is, the reconstructions will downloaded to each landmark folder automatically.

### Generating a Scene File

tool location: [`tools/prep_cambridge_2D3D_data.py`](tools/prep_cambridge_2D3D_data.py)

By default the tool assumes there is a folder with the undistorted SfM reconstruction of MegaDepth symlinked to the `data` folder, as explained in the previous section. If the folder is symlinked, theoretically no arguments need to passed. Just call the tool with
```
python tools/prep_cambridge_2D3D_data.py
```

After completion, the tool will generate a number of files in `data/cambridge/data_processed/query-netvlad10`.

### Generating a Descriptor Cache

Run the tool `tools/extract_features_immatch.py`.

```
python -m tools.extract_features_immatch --immatch_config 'immatch/superpoint'   --dataset 'cambridge'
```
This will generate the descriptor cache for SuperPoint.


## 7-Scenes
### Getting the Dataset

Setting up this dataset was originally a somewhat involved task, so we ended up creating a tool for downloading and extracting the dataset. It's in `tools/download_7scenes.py`. From the root folder of this project, call it as
```
python tools/download_7scenes.py <prefix>
```
where `<prefix>` is a folder where the new dataset will be downloaded to. The tool will create a folder inside it and place all the necessary contents inside `<prefix>/7scenes`. Don't forget to symlink to the `data` folder.



