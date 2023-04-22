import os
import sys
import shutil
import argparse
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from typing_extensions import Annotated, Literal

import colmap_utils
import process_data_utils
from process_data_utils import CAMERA_MODELS

CONSOLE = Console(width=120)


def check_ffmpeg_installed():
    """Checks if ffmpeg is installed."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        CONSOLE.print("[bold red]Could not find ffmpeg. Please install ffmpeg.")
        print("See https://ffmpeg.org/download.html for installation instructions.")
        print("ffmpeg is only necessary if using videos as input.")
        sys.exit(1)


def check_colmap_installed():
    """Checks if colmap is installed."""
    colmap_path = shutil.which("colmap")
    if colmap_path is None:
        CONSOLE.print("[bold red]Could not find COLMAP. Please install COLMAP.")
        print("See https://colmap.github.io/install.html for installation instructions.")
        sys.exit(1)


@dataclass
class ProcessImages:
    """Process images into a nerfstudio dataset.

    This script does the following:

    1. Scales images to a specified size.
    2. Calculates the camera poses for each image using `COLMAP <https://colmap.github.io/>`_.
    """

    data: Path
    """Path the data, either a video file or a directory of images."""
    output_dir: Path
    """Path to the output directory."""
    camera_type: Literal["perspective", "fisheye"] = "perspective"
    """Camera model to use."""
    matching_method: Literal["exhaustive", "sequential", "vocab_tree"] = "vocab_tree"
    """Feature matching method to use. Vocab tree is recommended for a balance of speed and
        accuracy. Exhaustive is slower but more accurate. Sequential is faster but should only be used for videos."""
    sfm_tool: Literal["any", "colmap", "hloc"] = "any"
    """Structure from motion tool to use. Colmap will use sift features, hloc can use many modern methods
       such as superpoint features and superglue matcher"""
    feature_type: Literal[
        "any",
        "sift",
        "superpoint",
        "superpoint_aachen",
        "superpoint_max",
        "superpoint_inloc",
        "r2d2",
        "d2net-ss",
        "sosnet",
        "disk",
    ] = "any"
    """Type of feature to use."""
    matcher_type: Literal[
        "any", "NN", "superglue", "superglue-fast", "NN-superpoint", "NN-ratio", "NN-mutual", "adalam"
    ] = "any"
    """Matching algorithm."""
    num_downscales: int = 3
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
        will downscale the images by 2x, 4x, and 8x."""
    skip_colmap: bool = False
    """If True, skips COLMAP and generates transforms.json if possible."""
    colmap_cmd: str = "colmap"
    """How to call the COLMAP executable."""
    mvs_dense: bool = False
    """If True, will do the patch match stereo"""
    gpu: bool = True
    """If True, use GPU."""
    verbose: bool = False
    """If True, print extra logging."""

    def main(self) -> None:
        """Process images into a nerfstudio dataset."""
        check_ffmpeg_installed()
        check_colmap_installed()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        summary_log = []

        # Copy images to output directory
        num_frames = process_data_utils.copy_images(self.data, image_dir=image_dir, verbose=self.verbose)
        summary_log.append(f"Starting with {num_frames} images")

        # Downscale images
        summary_log.append(process_data_utils.downscale_images(image_dir, self.num_downscales, verbose=self.verbose))

        # Run COLMAP
        colmap_dir = self.output_dir / "colmap"
        if not self.skip_colmap:
            colmap_dir.mkdir(parents=True, exist_ok=True)

            (sfm_tool, feature_type, matcher_type) = process_data_utils.find_tool_feature_matcher_combination(
                self.sfm_tool, self.feature_type, self.matcher_type
            )

            if sfm_tool == "colmap":
                colmap_utils.run_colmap(
                    image_dir=image_dir,
                    colmap_dir=colmap_dir,
                    camera_model=CAMERA_MODELS[self.camera_type],
                    gpu=self.gpu,
                    verbose=self.verbose,
                    matching_method=self.matching_method,
                    colmap_cmd=self.colmap_cmd,
                    mvs_dense=self.mvs_dense
                )
            else:
                CONSOLE.log("[bold red]Invalid combination of sfm_tool, feature_type, and matcher_type, exiting")
                sys.exit(1)

        # Save transforms.json
        if (colmap_dir / "sparse" / "0" / "cameras.bin").exists():
            with CONSOLE.status("[bold yellow]Saving results to transforms.json", spinner="balloon"):
                num_matched_frames = colmap_utils.colmap_to_json(
                    cameras_path=colmap_dir / "sparse" / "0" / "cameras.bin",
                    images_path=colmap_dir / "sparse" / "0" / "images.bin",
                    output_dir=self.output_dir,
                    camera_model=CAMERA_MODELS[self.camera_type],
                )
                summary_log.append(f"Colmap matched {num_matched_frames} images")
            summary_log.append(colmap_utils.get_matching_summary(num_frames, num_matched_frames))
        else:
            CONSOLE.log("[bold yellow]Warning: could not find existing COLMAP results. Not generating transforms.json")

        CONSOLE.rule("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.print(summary, justify="center")
        CONSOLE.rule()


@dataclass
class ProcessVideo:
    """Process videos into a nerfstudio dataset.

    This script does the following:

    1. Converts the video into images.
    2. Scales images to a specified size.
    3. Calculates the camera poses for each image using `COLMAP <https://colmap.github.io/>`_.
    """

    data: Path
    """Path the data, either a video file or a directory of images."""
    output_dir: Path
    """Path to the output directory."""
    num_frames_target: int = 300
    """Target number of frames to use for the dataset, results may not be exact."""
    camera_type: Literal["perspective", "fisheye"] = "perspective"
    """Camera model to use."""
    matching_method: Literal["exhaustive", "sequential", "vocab_tree"] = "vocab_tree"
    """Feature matching method to use. Vocab tree is recommended for a balance of speed and
        accuracy. Exhaustive is slower but more accurate. Sequential is faster but should only be used for videos."""
    sfm_tool: Literal["any", "colmap", "hloc"] = "colmap"
    """Structure from motion tool to use. Colmap will use sift features, hloc can use many modern methods
       such as superpoint features and superglue matcher"""
    feature_type: Literal[
        "any",
        "sift",
        "superpoint",
        "superpoint_aachen",
        "superpoint_max",
        "superpoint_inloc",
        "r2d2",
        "d2net-ss",
        "sosnet",
        "disk",
    ] = "sift"
    """Type of feature to use."""
    matcher_type: Literal[
        "any", "NN", "superglue", "superglue-fast", "NN-superpoint", "NN-ratio", "NN-mutual", "adalam"
    ] = "any"
    """Matching algorithm."""
    num_downscales: int = 2
    """Number of times to downscale the images. Downscales by 2 each time. For example a value of 3
        will downscale the images by 2x, 4x, and 8x."""
    skip_colmap: bool = False
    """If True, skips COLMAP and generates transforms.json if possible."""
    colmap_cmd: str = "colmap"
    """How to call the COLMAP executable."""
    mvs_dense: bool = False
    """If True, will do the patch match stereo"""
    gpu: bool = True
    """If True, use GPU."""
    verbose: bool = False
    """If True, print extra logging."""

    def main(self) -> None:
        """Process video into a nerfstudio dataset."""
        check_ffmpeg_installed()
        check_colmap_installed()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.output_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)

        # Convert video to images
        summary_log, num_extracted_frames = process_data_utils.convert_video_to_images(
            self.data, image_dir=image_dir, num_frames_target=self.num_frames_target, verbose=self.verbose
        )

        # Downscale images
        summary_log.append(process_data_utils.downscale_images(image_dir, self.num_downscales, verbose=self.verbose))

        # Run Colmap
        colmap_dir = self.output_dir / "colmap"
        if not self.skip_colmap:
            colmap_dir.mkdir(parents=True, exist_ok=True)

            (sfm_tool, feature_type, matcher_type) = process_data_utils.find_tool_feature_matcher_combination(
                self.sfm_tool, self.feature_type, self.matcher_type
            )

            if sfm_tool == "colmap":
                colmap_utils.run_colmap(
                    image_dir=image_dir,
                    colmap_dir=colmap_dir,
                    camera_model=CAMERA_MODELS[self.camera_type],
                    gpu=self.gpu,
                    verbose=self.verbose,
                    matching_method=self.matching_method,
                    colmap_cmd=self.colmap_cmd,
                    mvs_dense=self.mvs_dense,
                )
            else:
                CONSOLE.log("[bold red]Invalid combination of sfm_tool, feature_type, and matcher_type, exiting")
                sys.exit(1)
        
        num_matched_frames = colmap_utils.colmap_to_json(
                    cameras_path=colmap_dir / "sparse" / "0" / "cameras.bin",
                    images_path=colmap_dir / "sparse" / "0" / "images.bin",
                    points3D_path=colmap_dir / "sparse" / "0" / "points3D.bin",
                    output_dir=self.output_dir,
                    camera_model=CAMERA_MODELS[self.camera_type],
                )

        # Save transforms.json
        if (colmap_dir / "sparse" / "0" / "cameras.bin").exists():
            with CONSOLE.status("[bold yellow]Saving results", spinner="balloon"):
                num_matched_frames = colmap_utils.colmap_to_json(
                    cameras_path=colmap_dir / "sparse" / "0" / "cameras.bin",
                    images_path=colmap_dir / "sparse" / "0" / "images.bin",
                    points3D_path=colmap_dir / "sparse" / "0" / "points3D.bin",
                    output_dir=self.output_dir,
                    camera_model=CAMERA_MODELS[self.camera_type],
                )
                summary_log.append(f"Colmap matched {num_matched_frames} images")
            summary_log.append(colmap_utils.get_matching_summary(num_extracted_frames, num_matched_frames))
        else:
            CONSOLE.log("[bold yellow]Warning: could not find existing COLMAP results. Not generating pose infomations")

        CONSOLE.rule("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

        for summary in summary_log:
            CONSOLE.print(summary, justify="center")
        CONSOLE.rule()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        required=True,
        help="path to images/video",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="path to store output",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        required=True,
        help="gpu for cuda_visible_devices",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        assert len(args.gpu) == 1
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    datapath = os.path.abspath(args.data)
    outputpath = args.output_dir

    proc = None
    if os.path.isdir(datapath): 
        proc = ProcessImages(
            Path(datapath), Path(outputpath),
            num_downscales = 0,
            skip_colmap = False,
            mvs_dense=False,
            gpu = True,
            verbose  = False,
        )
    elif os.path.isfile(datapath) and (datapath[-3:] in ['mp4', 'csv', 'vid', 'ebm']):
        proc = ProcessVideo(
            Path(datapath), Path(outputpath),
            num_frames_target = 300,
            num_downscales = 0,
            skip_colmap = False,
            mvs_dense=True,
            gpu = True,
            verbose  = False,
            )
    else:
        print(f'please check the data path: {datapath}')
        exit()
    proc.main()
