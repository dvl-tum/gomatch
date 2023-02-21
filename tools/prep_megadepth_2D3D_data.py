from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
import glob
import os
from os.path import expanduser
import pickle
import yaml

from colmap.read_write_model import read_points3d_binary
import numpy as np
from tqdm import tqdm

from gomatch.utils.logger import get_logger
from PIL import Image

# For deterministic computation
np.random.seed(0)
logger = None


def process_scenes(args):

    global logger
    scene_info_dir = os.path.join(args.base_dir, "scene_info")

    # Init output dir & logger
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger.info(args)
    topk_min, topk_max = args.topk
    statis = Namespace(ovs=[], ndups=[], nims=[], nqs=[])
    data_dict = {}
    scene_files = sorted(glob.glob(os.path.join(scene_info_dir, "*.npz")))
    data_save_path = os.path.join(
        save_dir,
        f"megadepth_2d3d_q{args.scene_max}ov{args.overlap[0]}-{args.overlap[1]}covis{topk_min}-{topk_max}.npy",
    )
    logger.info(f"Save path: {data_save_path}")
    logger.info(f"Start process {len(scene_files)} scenes ......")

    for i, scene_file in enumerate(scene_files):
        scene_name = os.path.splitext(os.path.basename(scene_file))[0]

        # Load scene information
        try:
            scene_info = dict(np.load(scene_file, allow_pickle=True))
        except Exception as err:
            logger.info(f"Can not open {scene_file}: {err.args}")
            continue

        #  Load necessary scene information
        image_paths = scene_info["image_paths"]
        intrinsics = scene_info["intrinsics"]
        poses = scene_info["poses"]
        points3D_id_to_2D = scene_info["points3D_id_to_2D"]
        overlap_matrix = scene_info["overlap_matrix"]  # Already ignore diagnoal

        # Select valid pairs based on overlapping
        valid = np.logical_and(
            overlap_matrix >= args.overlap[0], overlap_matrix <= args.overlap[1]
        )
        valid_pair_ids = np.vstack(np.where(valid))  # 2,N
        valid_num = valid_pair_ids.shape[1]

        # Initialize scene data
        query_ids = []
        ims = {}
        ovs = []
        ndups = 0

        # Construct data for queries and topk dbs
        vals, ids, counts = np.unique(
            valid_pair_ids[0, :], return_index=True, return_counts=True
        )
        randomized = np.arange(len(vals))
        np.random.shuffle(randomized)
        for i in randomized:
            qid = vals[i]
            iid = ids[i]
            k = counts[i]
            if k < topk_min:
                continue

            # Make sure image uncorrupted
            name = image_paths[qid].replace("Undistorted_SfM/", "")
            im_path = os.path.join(args.base_dir, name)
            try:
                im = Image.open(im_path)
            except:
                logger.error(f"Can't open image {im_path}, skip!!")
                continue

            # Load query pose
            pose = poses[qid]
            R = pose[:3, :3]
            t = pose[:3, 3]
            K = intrinsics[qid]

            # Load 3D points
            pts3d2d = points3D_id_to_2D[qid]
            pts3d = np.array(list(pts3d2d.keys()))

            # Randomly select topk db images
            dbids = valid_pair_ids[1, iid : iid + k]
            np.random.shuffle(dbids)
            if len(dbids) > topk_max:
                dbids = dbids[:topk_max]

            # Save query data
            ims[qid] = Namespace(
                name=name,
                K=K,
                R=R,
                t=t,
                w=im.width,
                h=im.height,
                topk=dbids,
                ovs=[],
                pts3d=pts3d,
            )

            # Save db data
            ovs_ = ims[qid].ovs
            for dbid in dbids:
                ovs_.append(overlap_matrix[qid, dbid])
                ovs += ovs_
                if dbid not in ims:
                    pts3d2d = points3D_id_to_2D[dbid]
                    pose = poses[dbid]
                    R = pose[:3, :3]
                    t = pose[:3, 3]
                    K = intrinsics[dbid]
                    pts3d = np.array(list(pts3d2d.keys()))
                    name = image_paths[dbid].replace("Undistorted_SfM/", "")
                    im_path = os.path.join(args.base_dir, name)

                    # Check image size
                    try:
                        im = Image.open(im_path)
                    except:
                        logger.error(f"Can't open image {im_path}, skip!!")
                        continue
                    ims[dbid] = Namespace(
                        name=name, K=K, R=R, t=t, w=im.width, h=im.height, pts3d=pts3d
                    )
                else:
                    ndups += 1
            query_ids.append(qid)
            if len(query_ids) >= args.scene_max:
                break
        if len(query_ids) == 0:
            logger.info(
                f"Scene {scene_name} valid pairs {valid_num}, no queries, skip..."
            )
            continue
        scene_data = {"qids": query_ids, "ims": ims}

        # Record statis
        statis.ovs += ovs
        statis.ndups.append(ndups)
        statis.nqs.append(len(query_ids))
        statis.nims.append(len(ims))

        # Save data
        data_dict[scene_name] = scene_data
        np.save(data_save_path, data_dict)
        logger.info(f"Scene {scene_name}, qs={len(query_ids)} ims={len(ims)} ")

    logger.info(
        f"Finished process saved to {data_save_path}\n"
        f"Total used scenes={len(data_dict)} qs={np.sum(statis.nqs)} "
        f"dups={np.sum(statis.ndups)} ims={np.sum(statis.nims)} "
        f"3dpts={np.sum(statis.nims)}"
    )
    bins = np.arange(0.15, 1.01, 0.1)
    hist = np.histogram(statis.ovs, bins=bins)[0]
    logger.info(f"Overlappings :\nbins={bins}\nhist={hist}")

def cache_3d_data(args):
    global logger

    save_pickle = args.save_pickle
    ext =  'pkl' if save_pickle else 'npy'
    base_dir = args.base_dir
    save_dir = args.save_dir
    split_config = args.split_config

    # Load data splits: {split_tag: [scene_ids]}
    with open(split_config, 'r') as f:
        data_splits = yaml.load(f, Loader=yaml.FullLoader)['megadepth']['splits']
        logger.info(f"Data splits: {data_splits.keys()}")

    points3D_dir = os.path.join(save_dir, "scene_points3d")
    if not os.path.exists(points3D_dir):
        os.makedirs(points3D_dir)
    logger.info(
        f"Merge 3D data: target save dir: {points3D_dir} config: {split_config}"
    )

    # Save 3D points compactly for data splits
    for split in data_splits:
        save_path = os.path.join(points3D_dir, f"{split}.{ext}")
        if os.path.exists(save_path):
            logger.info(f"{save_path} existed.")
            continue

        data = {}
        scene_list = data_splits[split]
        for scene_name in tqdm(scene_list, total=len(scene_list)):
            pts3d_path = os.path.join(base_dir, scene_name, "sparse", "points3D.bin")
            if not os.path.exists(pts3d_path):
                logger.info(f"{pts3d_path} does not exist, skip!")
                continue
            scene_points3d = read_points3d_binary(pts3d_path)
            point3Ds = {
                v.id: np.concatenate([v.xyz, v.rgb, [v.error]])
                for v in scene_points3d.values()
            }
            data[scene_name] = point3Ds

        if save_pickle:
            with open(save_path, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            np.save(save_path, data)
        logger.info(f"Split={split} scenes={len(scene_list)}\nSaved to {save_path}")


def parse_arguments():

    parser = ArgumentParser(
        description="Command line tool to preprocess MegaDepth's data set into a condensed, lightweight format, tailored for blind-pnp.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        type=expanduser,
        default=os.path.join("data", "MegaDepth_undistort"),
        help="Path to the MegaDepth dataset folder",
    )
    parser.add_argument(
        "--save-dir",
        type=expanduser,
        default=os.path.join("data", "MegaDepth_undistort", "data_processed", "v2"),
        help="Output folder for the preprocessed dataset.",
    )
    parser.add_argument(
        "--log",
        default="megadepth_preprocess.log",
        help="File name for the log produced by the processing tool.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        nargs=2,
        default=(0.35, 1),
        help="Minimum point overlap or an overlap range between images that ensures selection. Range: [0.0, 1.0]",
    )
    parser.add_argument(
        "--topk",
        type=int,
        nargs=2,
        default=[3, 10],
        metavar="K",
        help="Minimum and maximum number of images to be retrieved for each query.",
    )
    parser.add_argument(
        "--scene-max",
        default=500,
        metavar="N",
        type=int,
        help="Maximum number of queries per scene.",
    )
    parser.add_argument(
        "--split-config",
        default=os.path.join("configs", "datasets.yml"),
        type=str,
        help="Dataset yaml config to define the data splits.",
    )
    parser.add_argument(
        "--save-pickle",
        action="store_true",
        help="If specified, save 3d data in pickle format. Otherwise in numpy format.",
    )
    parser.add_argument(
        "--skip-scene-parsing",
        action="store_true",
        help="Avanced usage: if specified, the first stage of processing is skipped. Use only if attempting to resume.",
    )
    parser.add_argument(
        "--skip-3d-data-processing",
        action="store_true",
        help="Avanced usage: if specified, skip processing the 3d point clouds.",
    )
    return parser.parse_args()


def main():

    global logger

    args = parse_arguments()

    # initialize singleton logger instance
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logger = get_logger(log_path=os.path.join(args.save_dir, args.log))

    if not args.skip_scene_parsing:
        process_scenes(args)

    if args.skip_3d_data_processing:
        return

    cache_3d_data(args)


if __name__ == "__main__":
    main()
