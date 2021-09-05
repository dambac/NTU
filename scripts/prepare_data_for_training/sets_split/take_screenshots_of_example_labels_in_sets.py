from pathlib import Path
from typing import List

import torch

from scripts.prepare_data_for_training.samples.samples_v2 import SampleV2
from scripts.utils.constants import C
from scripts.utils.frames import rgb_frame_to_image
from scripts.utils.serialization import read_json


def save_screens(samples: List[SampleV2]):
    dir0 = "outputs/samples_split_screenshots/0"
    dir1 = "outputs/samples_split_screenshots/1"

    Path(dir0).mkdir(exist_ok=True, parents=True)
    Path(dir1).mkdir(exist_ok=True, parents=True)

    sample: SampleV2
    for sample in samples:
        frame_name = sample.id
        frame_path = f"{C.Frames.FRAMES_SCREENSHOTS_PATH}/{frame_name}.pt"

        frame = torch.load(frame_path)[0]
        image = rgb_frame_to_image(frame)

        output_dir = dir0 if sample.label == 0 else dir1
        output_path = f"{output_dir}/{frame_name}.jpg"

        image.save(output_path)


def run():
    target_samples = "chart_images_5s_pure"
    target_video = "PH1011-PHYSICS_20150922"
    target_samples_split = "chart_images_5s_pure_by_video"
    target_set = "test"

    samples = SampleV2.from_csv(target_samples)
    samples_for_video = [s for s in samples if s.extract_video_name() == target_video]

    samples_split = read_json(f"{C.DistributionsAndSets.SETS_SPLITS_SAMPLES}/{target_samples_split}.json")
    samples_in_set = [s for s in samples_for_video if s.id in samples_split[target_set]]

    save_screens(samples_in_set)


if __name__ == "__main__":
    run()
