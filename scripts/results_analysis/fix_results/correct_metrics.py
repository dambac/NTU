import numpy as np

from analysis.anal_utils import load_results, save_results2
from scripts.definitions.results import RunResults, CombinationResults, IterationResults, FitMetrics, FitResults
from scripts.utils.common import zero_division_safe


class Correcter:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.results: RunResults = load_results(results_dir)

    def correct(self):
        comb_results: CombinationResults
        for comb_results in self.results.combination_results:

            iter_results: IterationResults
            for iter_results in comb_results.iterations_results:
                fit_results: FitResults = iter_results.fit_results

                fit_results.train_metrics = self.fix_metrics(fit_results.train_metrics)
                fit_results.dev_metrics = self.fix_metrics(fit_results.dev_metrics)
                fit_results.test_metrics = self.fix_metrics(fit_results.test_metrics)

        save_results2(self.results, self.results_dir)

    @staticmethod
    def fix_metrics(metrics: FitMetrics):
        c_00 = len(metrics.frames00)
        c_01 = len(metrics.frames01)
        c_10 = len(metrics.frames10)
        c_11 = len(metrics.frames11)

        precision0 = zero_division_safe(lambda _: c_00 / (c_00 + c_10))
        precision1 = zero_division_safe(lambda _: c_11 / (c_11 + c_01))
        precisions = np.array([precision0, precision1])

        recall0 = zero_division_safe(lambda _: c_00 / (c_00 + c_01))
        recall1 = zero_division_safe(lambda _: c_11 / (c_10 + c_11))
        recalls = np.array([recall0, recall1])

        up = 2 * (precisions * recalls)
        down = precisions + recalls
        f1_scores = np.divide(up, down, out=np.zeros_like(up), where=up != 0)

        return FitMetrics(acc=metrics.acc,
                          precisions=precisions.tolist(),
                          recalls=recalls.tolist(),
                          f1_scores=f1_scores.tolist(),
                          frames00=metrics.frames00,
                          frames01=metrics.frames01,
                          frames10=metrics.frames10,
                          frames11=metrics.frames11)
