from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
from glob import glob
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Union
from zipfile import ZipFile, BadZipFile

import wget

PathT = Union[str, Path]

SCENES = ["chess", "fire", "heads", "pumpkin", "redkitchen", "stairs"]
DATASET_URL = "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/{}.zip"
RETRIEVAL_URL = "https://cvg-data.inf.ethz.ch/pixloc_CVPR2021/7Scenes/7scenes_densevlad_retrieval/{}_top10.txt"
GRAVITY = dict(
    chess=["1.55444387e-02", "9.80374336e-01", "1.96531475e-01"],
    fire=["9.11076553e-03", "9.82454717e-01", "1.86279118e-01"],
    heads=["1.71073172e-02", "9.45489526e-01", "3.25202823e-01"],
    pumpkin=["4.65561338e-02", "9.58079576e-01", "2.82694459e-01"],
    redkitchen=["-8.87460355e-03", "9.04425621e-01", "4.26539183e-01"],
    stairs=["1.46721387e-02", "9.84022856e-01", "1.77436173e-01"],
)


def parse_arguments() -> Namespace:
    parser = ArgumentParser(
        description="Utility to assist in downloading and extracting all required data to run benchmarks on 7-scenes.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    abs_path = lambda x: Path(os.path.abspath(x))
    parser.add_argument(
        "prefix",
        type=abs_path,
        help="A new folder names '7scenes' will be created inside `prefix` with the dataset contents.",
    )
    return parser.parse_args()


def write_gravity_file(folder: PathT, data: Iterable) -> None:
    with open(os.path.join(folder, "gravity-direction.txt"), "w") as f:
        for d in data:
            f.write(f" {d}\n")


# for some reason extraction fails with most Thumbs.db files
_ignore_pattern = re.compile(r"Bad CRC-32 for file '.*Thumbs.db'")


def unzip(task: Callable[[], Any]) -> None:
    try:
        task()
    except BadZipFile as ex:
        if not _ignore_pattern.match(str(ex)):
            raise ex


def main() -> None:
    args = parse_arguments()

    # create directory
    dataset_folder = os.path.join(args.prefix, "7scenes")
    os.makedirs(dataset_folder, exist_ok=True)

    for scene in SCENES:
        # download
        url = DATASET_URL.format(scene)
        filename = wget.download(url, dataset_folder)

        def extract_scene():
            with ZipFile(filename, "r") as zip:
                zip.extractall(dataset_folder)

        unzip(extract_scene)

        # remove zip
        os.remove(filename)

        # download retrieval file
        scene_folder = os.path.join(dataset_folder, scene)
        wget.download(RETRIEVAL_URL.format(scene), scene_folder)

        # write gravity file
        write_gravity_file(scene_folder, GRAVITY[scene])

        # extract inner sequences
        for seq_zip in glob(os.path.join(scene_folder, "*.zip")):

            def extract_sequence():
                with ZipFile(seq_zip, "r") as zip:
                    zip.extractall(scene_folder)

            unzip(extract_sequence)
            os.remove(seq_zip)


if __name__ == "__main__":
    main()
