import re
from typing import List

from scripts.results_analysis.utils.results_analysis_utils import load_results
from scripts.results_post_processing.final_summary.load_results_summaries import get_iteration_results, \
    CHARTS_VGG_SUMMARIES_DIRS, WEBSITES_VGG_SUMMARIES_DIRS
from scripts.results_post_processing.final_summary.load_results_summaries import load_results_summaries
from scripts.results_post_processing.final_summary.results_def import ResultsSummary, IterationResults, get_model_name, \
    get_lr, \
    get_wd

# from final_summary.load_results import CHARTS_DIRS, load_results, get_iteration_results, WEBSITES_DIRS

NAMES_MAP = {
    "test_vgg_m_classic_adam": "basic",
    "test_vgg_m_classic_drop1_adam": "drop",
    "test_vgg_m_classic_drop_old_adam": "drop-part",
    "test_vgg_m_classic_less_layers_adam": "less-layers",
    "test_vgg_m_classic_less_layers_drop_adam": "less-layers-drop",
    "test_vgg_m_classic_less_nodes_adam": "less-nodes",
    "test_vgg_m_classic_small_both_drop_adam": "r-both",
    "test_vgg_m_classic_small_class_drop_adam": "r-class",
    "test_vgg_m_classic_small_feature_drop_adam": "r-feat",
    "test_vgg_m_classic_xavier_adam": "xavier",
}


def count_experiments_distribution(iteration_results: List[IterationResults], by):
    result = {}
    for res in iteration_results:
        key = by(res)
        if key not in result:
            result[key] = 0
        result[key] += 1
    result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
    return result


def analyze(input_dirs):
    results: List[ResultsSummary] = load_results(input_dirs)
    iteration_results: List[IterationResults] = get_iteration_results(results)

    iteration_results = sorted(iteration_results, key=lambda it_results: get_f1_sum_dev(it_results), reverse=True)
    best_results = iteration_results[:30]
    # IterationResults.to_df_multiple(best_results[:30], "output/websites/top30.csv")

    # experiments_distribution = count_experiments_distribution(best_results, by=lambda ir: ir.experiment_name)
    # models_distribution = count_experiments_distribution(best_results, by=lambda ir: get_model_name(ir.experiment_name))

    detailed_analysis(best_results)
    z = 1


def detailed_analysis(iteration_results: List[IterationResults]):
    top_10_by_model = count_experiments_distribution(iteration_results[:10],
                                                     by=lambda ir: get_model_name(ir.experiment_name))
    top_10_by_lr = count_experiments_distribution(iteration_results[:10],
                                                  by=lambda ir: get_lr(ir.experiment_name))
    top_10_by_wd = count_experiments_distribution(iteration_results[:10],
                                                  by=lambda ir: get_wd(ir.experiment_name))

    top_20_by_model = count_experiments_distribution(iteration_results[:20],
                                                     by=lambda ir: get_model_name(ir.experiment_name))
    top_20_by_lr = count_experiments_distribution(iteration_results[:20],
                                                  by=lambda ir: get_lr(ir.experiment_name))
    top_20_by_wd = count_experiments_distribution(iteration_results[:20],
                                                  by=lambda ir: get_wd(ir.experiment_name))

    top_30_by_model = count_experiments_distribution(iteration_results[:30],
                                                     by=lambda ir: get_model_name(ir.experiment_name))
    top_30_by_lr = count_experiments_distribution(iteration_results[:30],
                                                  by=lambda ir: get_lr(ir.experiment_name))
    top_30_by_wd = count_experiments_distribution(iteration_results[:30],
                                                  by=lambda ir: get_wd(ir.experiment_name))

    z = 1


def get_f1_sum_dev(iteration_results: IterationResults):
    return iteration_results.dev_results.f11 + iteration_results.dev_results.f10


def r5(value):
    return round(value, 5)


def get_experiment_pretty_name(name):
    file_name = re.search(".*(test.*)", name).group(1)

    for k, v in NAMES_MAP.items():
        if k in file_name:
            file_name = file_name.replace(k, v)
            break

    return file_name.replace("_", "\\_")


def print_top(dirs_with_experiments, set_attribute):
    results: List[ResultsSummary] = load_results_summaries(dirs_with_experiments)
    iteration_results: List[IterationResults] = get_iteration_results(results)

    iteration_results = sorted(iteration_results,
                               key=lambda it_results: getattr(it_results, set_attribute).f10 + getattr(it_results,
                                                                                                       set_attribute).f11,
                               reverse=True)

    best_10_subject_results_with_positions = get_top_10_subject_results(iteration_results)
    best_10_subject_results = [result[1] for result in best_10_subject_results_with_positions]
    best_subject_position = best_10_subject_results_with_positions[0][0]

    best_10_video_results = iteration_results[:10]
    worst_10_video_results_that_are_still_better_than_top1_subject = iteration_results[
                                                                     best_subject_position - 10:best_subject_position]

    for top_10_results in [best_10_video_results,
                           worst_10_video_results_that_are_still_better_than_top1_subject,
                           best_10_subject_results]:
        for ir in top_10_results:
            for sr in [ir.dev_results]:
                to_print = [
                    # ir.iteration,
                    get_experiment_pretty_name(ir.experiment_name),
                    sr.set_name,
                    r5(sr.acc),
                    r5(sr.f11),
                    r5(sr.rec1),
                    r5(sr.prec1),
                    r5(sr.f10),
                    r5(sr.rec0),
                    r5(sr.prec0),
                    r5(sr.f11 + sr.f10)
                    # get_split(ir.experiment_name)
                ]
                print(" & ".join([str(value) for value in to_print]) + "\\\\")
            print("\\hline")
        print()
        print()


def get_top_10_subject_results(iteration_results: List[IterationResults]):
    best_results = []
    for global_ranking, ir in enumerate(iteration_results):
        if "subject" in ir.experiment_name:

            if len(best_results) == 10:
                return best_results

            best_results.append((global_ranking, ir))
    return best_results


if __name__ == "__main__":
    # analyze(WEBSITES_DIRS)
    # print_top(CHARTS_DIRS)
    # print_top(dirs_with_experiments=CHARTS_VGG_SUMMARIES_DIRS, set_attribute="dev_results")
    print_top(dirs_with_experiments=WEBSITES_VGG_SUMMARIES_DIRS, set_attribute="dev_results")

    # print_top_train(WEBSITES_DIRS)

    # print_top_dev(["websites/transfer/test_transfers"])
    # analyze(CHARTS_DIRS)
