from typing import List

from final_summary.load_results import WEBSITES_DIRS, CHARTS_DIRS, load_results, get_iteration_results
from final_summary.results_def import *
import json as j

# dummy to not auto clean json import
j.dumps({})

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
    results: List[Results] = load_results(input_dirs)
    iteration_results: List[IterationResults] = get_iteration_results(results)

    iteration_results = sorted(iteration_results, key=lambda it_results: get_f1_sum_dev(it_results), reverse=True)
    failures = [ir for ir in iteration_results if is_failure(ir)]

    failures_by_experiment_name = count_experiments_distribution(failures,
                                                                 lambda ir: ir.experiment_name)
    failures_by_model_name = count_experiments_distribution(failures,
                                                            lambda ir: get_model_name(ir.experiment_name))
    failures_by_wd = count_experiments_distribution(failures,
                                                    lambda ir: get_wd(ir.experiment_name))
    failures_by_lr = count_experiments_distribution(failures,
                                                    lambda ir: get_lr(ir.experiment_name))
    failures_by_lrwd = count_experiments_distribution(failures,
                                                      lambda ir: get_lr(ir.experiment_name) + get_wd(
                                                          ir.experiment_name))
    z = 1

    best_results = iteration_results[:30]
    # IterationResults.to_df_multiple(best_results[:30], "output/websites/top30.csv")

    experiments_distribution = count_experiments_distribution(best_results, by=lambda ir: ir.experiment_name)
    models_distribution = count_experiments_distribution(best_results, by=lambda ir: get_model_name(ir.experiment_name))

    z = 1


def is_failure(it_results: IterationResults):
    set_results = [it_results.train_results]

    set_res: SetResults
    for set_res in set_results:

        not_zero = [set_res.rec1, set_res.prec1, set_res.f11]
        for value in not_zero:
            if value == 0:
                return True

    return False


def get_f1_sum_dev(iteration_results: IterationResults):
    return iteration_results.dev_results.f11 + iteration_results.dev_results.f10


if __name__ == "__main__":
    analyze(WEBSITES_DIRS)
    analyze(CHARTS_DIRS)
