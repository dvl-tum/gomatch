from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
import glob
import os
from os.path import expanduser

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
        save_dir, f"megadepth_2d3d_q{args.scene_max}ov{args.overlap[0]}-{args.overlap[1]}covis{topk_min}-{topk_max}.npy",
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
        valid = np.logical_and(overlap_matrix >= args.overlap[0], overlap_matrix <=  args.overlap[1])
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
            name = image_paths[qid].replace('Undistorted_SfM/', '')
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
            ims[qid] = Namespace(name=name, K=K, R=R, t=t, w=im.width, h=im.height,
                                 topk=dbids, ovs=[], pts3d=pts3d)

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
                    name = image_paths[dbid].replace('Undistorted_SfM/', '')
                    im_path = os.path.join(args.base_dir, name)

                    # Check image size
                    try:
                        im = Image.open(im_path)
                    except:
                        logger.error(f"Can't open image {im_path}, skip!!")
                        continue
                    ims[dbid] = Namespace(name=name, K=K, R=R, t=t, w=im.width, h=im.height, pts3d=pts3d)
                else:
                    ndups += 1
            query_ids.append(qid)
            if len(query_ids) >= args.scene_max:
                break
        if len(query_ids) == 0:
            logger.info(f'Scene {scene_name} valid pairs {valid_num}, no queries, skip...')
            continue
        scene_data = {'qids':query_ids, 'ims':ims}

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


def reduce_3d_data(base_dir, save_dir, split=False):
    global logger
    
    points3D_dir = os.path.join(save_dir, "scene_points3d")
    if not os.path.exists(points3D_dir):
        os.makedirs(points3D_dir)
    if not split:
        save_path = os.path.join(points3D_dir, "all.npy")
        if os.path.exists(save_path):
            logger.info(f"Finished, {save_path} existed.")
            return
        
    scene_info_dir = os.path.join(base_dir, "scene_info")
    scene_files = sorted(glob.glob(os.path.join(scene_info_dir, "*.npz")))
    logger.info(
        f"Reduce 3D data: target save dir: {points3D_dir} scenes to process: {len(scene_files)}"
    )

    # Save 3D points compactly in separate files
    data = dict()
    for scene_file in tqdm(scene_files):
        scene_name = os.path.splitext(os.path.basename(scene_file))[0]
        scene_points3d = read_points3d_binary(
            os.path.join(base_dir, scene_name, "sparse", "points3D.bin")
        )
        point3Ds = {
            v.id: np.concatenate([v.xyz, v.rgb, [v.error]])
            for v in scene_points3d.values()
        }

        if not split:
            data[scene_name] = point3Ds
            continue

        save_path = os.path.join(points3D_dir, f"{scene_name}.npy")
        if os.path.exists(save_path):
            logger.info(f"Skip {scene_name}.npy, existed.")
        np.save(save_path, point3Ds)
        logger.info(f"Save {scene_name}.npy, pts: {len(point3Ds)}")

    if not split:
        np.save(save_path, data)
        logger.info(f"Saved {save_path}")        


def split_3d_to_train_val(points3d_file):
    val_split = ['0024', '0021', '0025', '1589', '0019',
                 '0008', '0032', '0063', '0015', '0022',
                 '0044', '0087', '0060', '0183', '0389',
                 '0058', '0102', '0238', '0559', '0189',
                 '0446', '0024', '0107', '5004', '0107']

    points3d = np.load(points3d_file, allow_pickle=True).item()
    train = {}
    val = {}

    for sid in points3d:
        if sid in val_split:
            val[sid] = points3d[sid]
        else:
            train[sid] = points3d[sid]
    np.save(points3d_file.replace('all', 'train'), train)
    np.save(points3d_file.replace('all', 'val'), val)

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
        default=[3, 15],
        metavar="K",
        help="Minimum and maximum number of images to be retrieved for each query.",
    )
    parser.add_argument(
        "--scene-max",
        default=50,
        metavar="N",
        type=int,
        help="Maximum number of queries per scene.",
    )
    parser.add_argument(
        "--split-scenes",
        action="store_true",
        help="If specified, 3D points are stored in separate files per scene.",
    )
    parser.add_argument(
        "--split-train-val",
        action="store_true",
        help="If specified, 3D points are split into train val files.",
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

    reduce_3d_data(args.base_dir, args.save_dir, args.split_scenes)

    if not args.split_scenes and args.split_train_val:
        points3d_file = os.path.join(args.save_dir, "scene_points3d/all.npy")
        split_3d_to_train_val(points3d_file)

if __name__ == "__main__":
    main()