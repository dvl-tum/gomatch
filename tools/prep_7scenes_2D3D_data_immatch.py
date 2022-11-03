from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from collections import defaultdict
from glob import glob
import os
from os.path import expanduser
from pathlib import Path
import re
from typing import Iterable, Tuple
import torch

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from gomatch.utils.logger import get_logger


SCENES = ("chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs")
Kd = np.array([[585, 0, 320], [0, 585, 240], [0, 0, 1]])
Kc = np.array([[525, 0, 310], [0, 525, 234], [0, 0, 1]])


def parse_arguments():

    parser = ArgumentParser(
        description="Command line tool to preprocess 7scenes's data set into a condensed, lightweight format, tailored for our benchmark loader.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        type=expanduser,
        default=os.path.join("data", "7scenes"),
        help="Path to the MegaDepth dataset folder",
    )
    parser.add_argument(
        "--save-dir",
        type=expanduser,
        default=os.path.join("data", "7scenes", "data_processed_v2"),
        help="Output folder for the preprocessed dataset.",
    )
    parser.add_argument(
        "--log",
        default="7scenes_preprocess.log",
        help="File name for the log produced by the processing tool.",
    )
    parser.add_argument(
        "--frame-gap",
        type=int,
        default=10,
        help="Minimum guaranteed distance between query and retrieval images in the sequences",
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
        default=100,
        metavar="N",
        type=int,
        help="Maximum number of queries per scene.",
    )
    parser.add_argument(
        "--detector",
        default="sift",
        choices=["sift", "superpoint"],
        help="The type of keypoint detector used to select points of interest in the image.",
    )
    parser.add_argument(
        "--skip-scene-parsing",
        action="store_true",
        help="Avanced usage: if specified, the first stage of processing is skipped. Use only if attempting to resume.",
    )
    return parser.parse_args()


def init_detector(args):
    from immatch.utils.model_helper import init_model

    detector, _ = init_model(
        config=f"immatch/{args.detector}", benchmark_name="default", root_dir="."
    )
    return detector


def sample_query_ids(prefix, max_queries):
    total = len(list(glob(os.path.join(prefix, "*.pose.txt"))))
    max_queries = min(max_queries, total)

    # find section length
    length = int(np.ceil(total / max_queries))
    sizes = np.full(max_queries, length)
    sizes[-1] -= np.sum(sizes) - total

    # sample an id from each bucket
    offset = np.random.randint(length, size=max_queries) % sizes
    ids = np.array([0, *np.cumsum(sizes[:-1])]) + offset
    return ids


def generate_retrieval_ids(
    prefix: str, qids: Iterable[int], topk: Tuple[int, int], frame_gap: int
):
    # total number of frames
    total = len(list(glob(os.path.join(prefix, "*.pose.txt"))))

    retrieval_ids = []
    increments = np.arange(1, topk[1] + 1)
    diff = frame_gap * np.stack([increments, -increments], axis=-1).ravel()
    for q in qids:
        idx = q + diff
        mask = np.logical_and(idx >= 0, idx < total)
        retrieval_ids.append(idx[mask][: topk[1]])
    return retrieval_ids


# gravity align dataset
def gravity_align(g):
    g /= np.linalg.norm(g)
    axis = np.cross(g, (0, 1, 0))
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.dot(g, (0, 1, 0)))
    R = cv2.Rodrigues(axis * angle)[0]
    return R


def extract_pose(prefix, frame_id):
    # fetch original pose
    path = os.path.join(prefix, f"frame-{frame_id:06d}.pose.txt")
    pose = np.loadtxt(path)
    R = pose[:3, :3]
    t = pose[:3, -1]

    # apply gravity correction so that Y points down
    path = os.path.join(Path(prefix).parent, "gravity-direction.txt")
    gravity = np.loadtxt(path)
    Rg = gravity_align(gravity)

    # Correct for gravity
    Rc = Rg @ R
    tc = Rg @ t
    return Rc, tc


def extract_color_image_data(prefix, frame_id, detector):
    # Load image info
    name = os.path.join(prefix, f"frame-{frame_id:06d}.color.png")
    img = np.array(Image.open(name))
    height, width = img.shape[:2]

    # Extract keypoints
    kpts, _ = detector.load_and_extract(name)
    kpts = kpts.cpu().data.numpy() if isinstance(kpts, torch.Tensor) else kpts
    kpts = np.unique(kpts, axis=0)
    #     print(kpts.shape)
    # fix name to only retain info about seq and below
    name = os.path.join(*Path(name).parts[-3:])
    image = dict(name=name, width=width, height=height, kpts=kpts)
    return image


def extract_point_cloud(prefix, frame_id, pts2d_c):
    """pts2d_c assumes the 2d keypoints are coming from the color camera"""
    name = os.path.join(prefix, f"frame-{frame_id:06d}.depth.png")
    depth = np.array(Image.open(name)).astype(float)

    depth /= 1000.0  # mm to meter
    depth[(depth == 0.0) | (depth > 1000.0)] = np.nan  # filter invalid depth pixels

    # Map the points in color pixels to depth pixels
    Kc2d = Kd @ np.linalg.inv(Kc)
    pts2d_d = cv2.perspectiveTransform(pts2d_c.reshape(-1, 1, 2), Kc2d).squeeze()

    # find valid pixels inside the image
    h, w = depth.shape[:2]
    pts2d_di = pts2d_d.astype(int)
    mask = np.all(np.concatenate([pts2d_di >= 0, pts2d_di < (w, h)], axis=-1), axis=-1)
    pts2d_d, pts2d_di = pts2d_d[mask], pts2d_di[mask]

    # filter out pts with no depth
    depth_kp = depth[pts2d_di[:, 1], pts2d_di[:, 0]]
    mask = np.isfinite(depth_kp)
    pts2d_d, depth_kp = pts2d_d[mask], depth_kp[mask]

    # Uplift bearing vectors to 3d space
    pts2d_d1 = np.concatenate([pts2d_d, np.ones((len(pts2d_d), 1))], axis=-1)
    b2d = np.linalg.solve(Kd, pts2d_d1.T).T
    pts3d = b2d * depth_kp[:, None]
    return pts3d


def extract_frame_data(prefix, frame_id, detector):

    # extract all info from color image
    image = extract_color_image_data(prefix, frame_id, detector)

    # Build 3d point cloud in world space
    pts3d = extract_point_cloud(prefix, frame_id, image["kpts"])
    Rw, tw = extract_pose(prefix, frame_id)
    pts3d_w = pts3d @ Rw.T + tw

    # Convert pose to camera space
    Rc = Rw.T
    tc = -Rc @ tw

    frame = Namespace(
        K=Kc,
        name=image["name"],
        w=image["width"],
        h=image["height"],
        pts3d=pts3d_w,
        R=Rc,
        t=tc,
        topk=None,
    )
    return frame


def compute_total_sequences(prefix):
    nr_sequences = len(
        [
            seq
            for scene in SCENES
            for seq in glob(os.path.join(prefix, scene, "seq-[0-9][0-9]"))
        ]
    )
    return nr_sequences


def process_scenes(args):

    # maintenance
    nr_sequences = compute_total_sequences(args.base_dir)
    progress = tqdm(total=nr_sequences, unit="sequences")
    data_dict = dict()
    pts3d_data = dict()

    data_save_path = os.path.join(
        args.save_dir,
        f"7scenes_2d3d_q{args.scene_max}fg{args.frame_gap}"
        f"tp{args.topk[0]}-{args.topk[1]}"
        f"det{args.detector}1024.npy",
    )
    print(f"data_save_path:{data_save_path}")

    # directory maintenance
    pts3d_prefix = os.path.join(args.save_dir, "scene_points3d")
    os.makedirs(pts3d_prefix, exist_ok=True)

    # Initialize detector
    detector = init_detector(args)
    print(f">>>> Initialized detector {detector.name}")
    for scene in SCENES:
        pattern = os.path.join(args.base_dir, scene, "seq-[0-9][0-9]")
        for folder in glob(pattern):
            seq_id = "/".join(folder.split("/")[-2:])

            # Use scene max to split sequence in equally sized chunks
            # and sample a query image id from there.
            query_ids = sample_query_ids(folder, args.scene_max)

            # fetch retrieval images around id.
            retrieval_ids = generate_retrieval_ids(
                folder, query_ids, args.topk, args.frame_gap
            )

            # pop query if the minimum retrival frames are not guaranteed
            retrieval_dict = dict(zip(query_ids, retrieval_ids))
            for k, v in retrieval_dict.items():
                if len(v) < args.topk[0]:
                    del retrieval_dict[k]

            # collect final set of frame ids for which we need info
            frame_ids = set(retrieval_dict)
            for rids_i in retrieval_dict.values():
                frame_ids.update(rids_i)

            # extract all important information from each required frames
            pts3d_data[seq_id] = dict()
            ims = dict()
            print(f">>Process folder {folder} frames: {len(frame_ids)} ")
            for fid in frame_ids:
                frame = extract_frame_data(folder, fid, detector)

                # update pts3d information to conform with the data interfaces
                # we assume 3d points are only visible from a single frame
                n = len(pts3d_data[seq_id])
                pts3d_i = {n + i: pt for i, pt in enumerate(frame.pts3d)}
                pts3d_data[seq_id].update(pts3d_i)

                # update pts3d to only include ids
                frame.pts3d = np.array(list(pts3d_i.keys()))

                # update retrievals if it's a query id
                if fid in retrieval_dict:
                    frame.topk = retrieval_dict[fid]

                ims[fid] = frame

            # store everything
            data_dict[seq_id] = dict(qids=list(retrieval_dict), ims=ims)
            progress.update()

    progress.close()

    # save everything
    np.save(data_save_path, data_dict)
    np.save(os.path.join(pts3d_prefix, "test_all.npy"), pts3d_data)


def parse_queries_and_retrievals(filepath):

    pairs = defaultdict(list)
    with open(filepath) as f:
        for line in f.readlines():
            query_id, retrieval_id = line.rstrip().split()
            query_id = query_id.replace(".color.png", "").replace("frame-", "")
            retrieval_id = retrieval_id.replace(".color.png", "").replace("frame-", "")
            pairs[query_id].append(retrieval_id)
    return pairs


def process_scenes_retrieval(args):
    data_dict = dict()
    pts3d_data = dict()

    # Initialize detector
    detector = init_detector(args)
    print(f">>>> Initialized detector {detector.name}")

    for scene in SCENES:
        pts3d_data[scene] = dict()
        ims = dict()

        scene_folder = os.path.join(args.base_dir, scene)
        retrieval_dict = parse_queries_and_retrievals(
            os.path.join(scene_folder, f"{scene}_top10.txt")
        )

        # collect final set of frame ids for which we need info
        frame_ids = set(retrieval_dict)
        for rids_i in retrieval_dict.values():
            frame_ids.update(rids_i)

        print(f">>Process folder {scene_folder} frames: {len(frame_ids)} ")
        for fid_full in tqdm(sorted(frame_ids), unit="frames"):
            seq_id, fid = fid_full.split("/")
            fid = int(fid)

            seq_folder = os.path.join(scene_folder, seq_id)
            frame = extract_frame_data(seq_folder, fid, detector)

            # update pts3d information to conform with the data interfaces
            # we assume 3d points are only visible from a single frame
            n = len(pts3d_data[scene])
            pts3d_i = {n + i: pt for i, pt in enumerate(frame.pts3d)}
            pts3d_data[scene].update(pts3d_i)

            # update pts3d to only include ids
            frame.pts3d = np.array(list(pts3d_i.keys()))

            # update retrievals if it's a query id
            if fid_full in retrieval_dict:
                frame.topk = retrieval_dict[fid_full]

            ims[fid_full] = frame

        # store everything
        data_dict[scene] = dict(qids=list(retrieval_dict), ims=ims)

    # save everything
    data_save_path = os.path.join(
        args.save_dir, f"densevlad-top10-{args.detector}", "7scenes_2d3d.npy"
    )
    pts3d_prefix = os.path.join(
        args.save_dir, f"densevlad-top10-{args.detector}", "scene_points3d"
    )
    os.makedirs(pts3d_prefix, exist_ok=True)
    np.save(data_save_path, data_dict)
    np.save(os.path.join(pts3d_prefix, "all.npy"), pts3d_data)
    os.symlink(
        os.path.join(pts3d_prefix, "all.npy"), os.path.join(pts3d_prefix, "test.npy")
    )


def main():

    global logger

    args = parse_arguments()
    #     args.save_dir = os.path.join(args.save_dir, args.detector)

    # initialize singleton logger instance
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logger = get_logger(log_path=os.path.join(args.save_dir, args.log))

    process_scenes_retrieval(args)


#     process_scenes(args)


if __name__ == "__main__":
    main()
