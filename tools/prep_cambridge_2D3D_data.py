from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from collections import defaultdict
import os
from os.path import expanduser
from warnings import warn

from colmap.read_write_model import read_model, qvec2rotmat
import numpy as np
from tqdm import tqdm

from gomatch.utils.geometry import camera_params_to_intrinsics_mat
from gomatch.utils.logger import get_logger


SCENES = ("GreatCourt", "KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch")


def parse_arguments():

    parser = ArgumentParser(
        description="Command line tool to preprocess Cambridge Landmark's data set into a condensed, lightweight format, tailored for our benchmark loader.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        type=expanduser,
        default=os.path.join("data", "cambridge"),
        help="Path to the Cambridge dataset folder",
    )
    parser.add_argument(
        "--save-dir",
        type=expanduser,
        default=os.path.join("data", "cambridge", "data_processed"),
        help="Output folder for the preprocessed dataset.",
    )
    parser.add_argument(
        "--log",
        default="cambridge_preprocess.log",
        help="File name for the log produced by the processing tool.",
    )
    parser.add_argument(
        "--pairs",
        default="query-netvlad10",
        choices=[
            "oracle-10deg-10",
            "oracle-30deg-10",
            "oracle-covis-10",
            "query-netvlad10",
        ],
        help="Supported query-retrieval pairs this dataset supports.",
    )
    return parser.parse_args()


def parse_queries_and_retrievals(prefix, pairs_opt):

    pairs = defaultdict(list)
    with open(os.path.join(prefix, f"pairs-{pairs_opt}.txt")) as f:
        for line in f.readlines():
            query_id, retrieval_id = line.rstrip().split()
            pairs[query_id].append(retrieval_id)
    return pairs


def parse_dataset_poses(prefix):

    ds_poses = dict()  # key: frame-id, value: fields R and t
    for split in ("train", "test"):
        with open(os.path.join(prefix, f"dataset_{split}.txt")) as f:
            for line_n, line in enumerate(f):
                # skip headers
                if line_n < 3:
                    continue

                tokens = line.rstrip().split()
                frame_id = tokens[0]
                R = qvec2rotmat(np.array(list(map(float, tokens[4:]))))
                t = -R @ np.array(list(map(float, tokens[1:4])))
                ds_poses[frame_id] = dict(R=R, t=t)
    return ds_poses


def parse_dataset_intrinsics(prefix):
    ds_intrinsics = dict()
    with open(os.path.join(prefix, "query_list_with_intrinsics.txt")) as f:
        for line in f:
            tokens = line.rstrip().split()
            ds_intrinsics[tokens[0]] = dict(
                model=tokens[1],
                width=int(tokens[2]),
                height=int(tokens[3]),
                params=np.array(list(map(float, tokens[4:]))),
            )
    return ds_intrinsics


def process_scenes(args):
    """
    Variables prefixed with:
    ds_ - come from the original dataset
    col_ - come from a colmap sfm reconstruction using superpoint + superglue
    """
    global logger

    # maintenance
    data_dict = dict()
    pts3d_data = dict()

    for scene in tqdm(SCENES, unit="scenes"):
        scene_dir = os.path.join(args.base_dir, scene)

        # parse pairs
        pairs = parse_queries_and_retrievals(scene_dir, args.pairs)

        # parse dataset poses
        poses = parse_dataset_poses(scene_dir)

        # parse dataset instrinsics
        intrinsics = parse_dataset_intrinsics(scene_dir)

        # load colmap data
        col_cameras, col_images, col_points_3d = read_model(
            os.path.join(scene_dir, "sfm_superpoint+superglue"), ext=".bin"
        )

        # update original dataset with colmap data
        query_to_image_id = {image.name: image.id for image in col_images.values()}
        for qid, iid in query_to_image_id.items():
            img = col_images[iid]
            poses[qid] = dict(R=qvec2rotmat(img.qvec), t=img.tvec)
            intrinsics[qid] = {
                k: getattr(col_cameras[img.camera_id], k)
                for k in ("model", "width", "height", "params")
            }

        # Generate frame data
        # collect final set of frame ids for which we need info
        frame_ids = set(pairs)
        for rids_i in pairs.values():
            frame_ids.update(rids_i)

        ims_data = dict()
        for fid in frame_ids:

            pts3d = None
            if fid in query_to_image_id:
                img = col_images[query_to_image_id[fid]]
                pts3d = img.point3D_ids[img.point3D_ids >= 0]

            if fid not in intrinsics:
                msg = (
                    f"Frame {scene}/{fid} has no intrinsics available. "
                    "Removing it from the list of valid queries"
                )
                logger.warning(msg)
                del pairs[fid]
                continue

            cam = intrinsics[fid]
            ims_data[fid] = Namespace(
                name=os.path.join(scene, fid),
                w=cam["width"],
                h=cam["height"],
                K=camera_params_to_intrinsics_mat(cam),
                radial=cam["params"][-1],
                pts3d=pts3d,
                R=poses[fid]["R"],
                t=poses[fid]["t"],
                topk=pairs[fid] if fid in pairs else None,
            )

        # Populate qids
        data_dict[scene] = dict(qids=list(pairs), ims=ims_data)

        # 3d points
        pts3d_data[scene] = {
            v.id: np.concatenate([v.xyz, v.rgb, [v.error]])
            for v in col_points_3d.values()
        }

    # commit to storage
    save_prefix = os.path.join(args.save_dir, args.pairs)

    # 3d point data
    pts3d_prefix = os.path.join(save_prefix, "scene_points3d")
    os.makedirs(pts3d_prefix, exist_ok=True)
    np.save(os.path.join(pts3d_prefix, "all.npy"), pts3d_data)

    # scene file
    data_save_path = os.path.join(save_prefix, "cambridge_2d3d.npy")
    np.save(data_save_path, data_dict)


def main():

    global logger

    args = parse_arguments()

    # initialize singleton logger instance
    save_prefix = os.path.join(args.save_dir, args.pairs)
    if not os.path.exists(save_prefix):
        os.makedirs(save_prefix)
    logger = get_logger(log_path=os.path.join(save_prefix, args.log))

    process_scenes(args)


if __name__ == "__main__":
    main()
