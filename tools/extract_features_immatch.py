import argparse
from argparse import Namespace
import os
import numpy as np
from tqdm import tqdm
import time
import os
import torch
from pathlib import Path
import yaml
from immatch.utils.model_helper import init_model

cambridge_splits = dict(
    test=["KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch"],
    kings=["KingsCollege"],
    old=["OldHospital"],
    shop=["ShopFacade"],
    stmarys=["StMarysChurch"],
)


def compute_scene_im_features(detector, root_dir, dataset, split, dataset_conf):
    with open(dataset_conf, "r") as f:
        dataset_conf = Namespace(**yaml.load(f, Loader=yaml.FullLoader)[dataset])
    im_dir = Path(root_dir) / "data" / dataset_conf.data_dir
    data_processed_dir = Path(root_dir) / "data" / dataset_conf.data_processed_dir
    data_file = data_processed_dir / dataset_conf.data_file
    sids_to_load = dataset_conf.splits[split]

    # Initialize detector
    feature_cache_dir = data_processed_dir / "desc_cache" / detector.name
    feature_cache_dir.mkdir(exist_ok=True, parents=True)

    # Identify scene ids for the target split
    print(f">>>>Loading data from  {data_file} ")
    data_dict = np.load(data_file, allow_pickle=True).item()

    # Load all query ids, scene ids and image data
    print(
        f"Extract features per scene, method={detector.name} cache dir={feature_cache_dir}"
    )
    for sid in tqdm(sids_to_load, total=len(sids_to_load)):
        print(sid)
        if sid not in data_dict:
            continue

        feature_path = feature_cache_dir / f"{sid}.npy"
        scene_qids = data_dict[sid]["qids"]
        scene_ims = data_dict[sid]["ims"]
        updated_count = 0
        if feature_path.exists():
            scene_features = np.load(feature_path, allow_pickle=True).item()
        else:
            scene_features = {}
        print(f"sid={sid} qids={len(scene_qids)} ims={len(scene_ims)}")

        # Extract features
        for imid, im in scene_ims.items():
            if imid not in scene_features:
                im_path = os.path.join(im_dir, im.name)
                kpts, descs = detector.load_and_extract(im_path)
                kpts = (
                    kpts.cpu().data.numpy() if isinstance(kpts, torch.Tensor) else kpts
                )
                descs = (
                    descs.cpu().data.numpy()
                    if isinstance(descs, torch.Tensor)
                    else descs
                )
                scene_features[imid] = {"kpts": kpts, "descs": descs}
                updated_count += 1
            im.kpts = scene_features[imid]["kpts"]
            im.descs = scene_features[imid]["descs"]

        if updated_count > 0:
            print(f"Save {updated_count} new image features. ")
            Path(feature_path).parent.mkdir(exist_ok=True, parents=True)
            np.save(feature_path, scene_features)


def extract_and_save(args):
    detector, _ = init_model(
        config=args.immatch_config, benchmark_name="default", root_dir=args.root_dir
    )
    compute_scene_im_features(
        detector, args.root_dir, args.dataset, args.split, args.dataset_config
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config", type=str, default="configs/datasets.yml")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--immatch_config", type=str, default="sift")
    parser.add_argument("--dataset", type=str, default="megadepth")
    args = parser.parse_args()
    print(args)
    extract_and_save(args)
