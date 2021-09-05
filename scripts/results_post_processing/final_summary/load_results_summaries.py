import os
from typing import List

from scripts.results_post_processing.final_summary.results_def import ResultsSummary
from scripts.utils.constants import C

CHARTS_VGG_SUMMARIES_DIRS = [
    f"{C.RESULTS_ANALYSIS_DIR}/charts/model_experiments_subject",
    f"{C.RESULTS_ANALYSIS_DIR}/charts/model_experiments_video"
]

WEBSITES_VGG_SUMMARIES_DIRS = [
    f"{C.RESULTS_ANALYSIS_DIR}/websites/model_experiments_subject",
    f"{C.RESULTS_ANALYSIS_DIR}/websites/model_experiments_video"
]


def get_iteration_results(results: List[ResultsSummary]):
    iteration_results = []
    for result in results:
        iteration_results.extend(result.iteration_results)
    return iteration_results


def load_results_summaries(experiment_results_summaries_dir):
    single_results: List[ResultsSummary] = []

    for single_experiment_results_summary_dir in experiment_results_summaries_dir:

        dirs_with_summaries = os.listdir(single_experiment_results_summary_dir)

        for dir_with_summary in dirs_with_summaries:
            summary_file = f"{single_experiment_results_summary_dir}/{dir_with_summary}/summary/summary.csv"

            results = ResultsSummary.load(file=summary_file,
                                          experiment_name=f"{single_experiment_results_summary_dir}/{dir_with_summary}")
            single_results.append(results)

    return single_results
