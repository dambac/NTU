from pathlib import Path

import torch

from analysis.anal_utils import load_results
from scripts.definitions.results import *
from scripts.utils.common import create_dirs
from scripts.utils.constants import C
from scripts.utils.frames import rgb_frame_to_image


class Analyzer:
    def __init__(self, results_dir, frames_limit_dict):
        self.results: RunResults = load_results(results_dir)
        self.frames_limit_dict = frames_limit_dict

        self.base_output_dir = f"outputs/{results_dir}/frames"
        Path(self.base_output_dir).mkdir(exist_ok=True, parents=True)

    def analyze(self, iteration, combination=None):
        output_dir = f"{self.base_output_dir}/it{iteration}"
        Path(output_dir).mkdir(exist_ok=True, parents=True)

        combination_results: CombinationResults = [cr for cr in self.results.combination_results
                                                   if cr.combination.equals(combination)][0]
        iteration_results: IterationResults = combination_results.iterations_results[iteration]
        fit_results: FitResults = iteration_results.fit_results

        self.analyze_for_set(output_dir, "train", fit_results.train_metrics)
        self.analyze_for_set(output_dir, "dev", fit_results.dev_metrics)
        self.analyze_for_set(output_dir, "test", fit_results.test_metrics)

    def analyze_for_set(self, output_dir, set_name, fit_metrics: FitMetrics):
        set_output_dir = f"{output_dir}/{set_name}"

        output_00 = f"{set_output_dir}/00"
        output_01 = f"{set_output_dir}/01"
        output_10 = f"{set_output_dir}/10"
        output_11 = f"{set_output_dir}/11"
        create_dirs([output_00, output_01, output_10, output_11])

        self.analyze_for_pred(output_00, fit_metrics.frames00, self.frames_limit_dict["00"])
        self.analyze_for_pred(output_01, fit_metrics.frames01, self.frames_limit_dict["01"])
        self.analyze_for_pred(output_10, fit_metrics.frames10, self.frames_limit_dict["10"])
        self.analyze_for_pred(output_11, fit_metrics.frames11, self.frames_limit_dict["11"])

    def analyze_for_pred(self, output_dir, frames, limit):
        for frame_name in frames[:limit]:
            frame_path = f"{C.Frames.FRAMES_SCREENSHOTS_PATH}/{frame_name}.pt"
            frame = torch.load(frame_path)[0]
            frame_image = rgb_frame_to_image(frame)

            image_path = f"{output_dir}/{frame_name}.jpg"
            frame_image.save(image_path)
