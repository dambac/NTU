from pathlib import Path
from typing import List

import pandas as pd

from analysis.anal_utils import load_results
from scripts.definitions.results import RunResults, CombinationResults, IterationResults, FitResults, FitMetrics


class Analyzer:
    def __init__(self, results_dir):
        self.results: RunResults = load_results(results_dir)

        output_dir = f"outputs/{results_dir}/summary"
        Path(output_dir).mkdir(exist_ok=True, parents=True)

        self.output_path = f"{output_dir}/summary.csv"

    def analyze(self):
        df_data = []

        comb_results: List[CombinationResults] = self.results.combination_results
        for comb_res in comb_results:
            t_model = comb_res.combination.t_model

            iter_results: List[IterationResults] = comb_res.iterations_results
            for iter_res in iter_results:
                fit_results: FitResults = iter_res.fit_results

                df_data.append(self.get_set_metrics(t_model, iter_res.iteration, "Train", fit_results.train_metrics))
                df_data.append(self.get_set_metrics(t_model, iter_res.iteration, "Dev", fit_results.dev_metrics))
                df_data.append(self.get_set_metrics(t_model, iter_res.iteration, "Test", fit_results.test_metrics))

        df = pd.DataFrame(df_data, columns=['Model', 'Iteration', 'Set',
                                            'Acc',
                                            'Prec 1', 'Rec 1', "F1 1",
                                            'Prec 0', 'Rec 0', "F1 0",
                                            ])
        df.to_csv(self.output_path, index=False)
        z = 1

    @staticmethod
    def get_set_metrics(model, iteration, set_name, set_metrics: FitMetrics):
        acc = set_metrics.acc

        prec_1 = set_metrics.precisions[1]
        recall_1 = set_metrics.recalls[1]
        f1_1 = set_metrics.f1_scores[1]

        prec_0 = set_metrics.precisions[0]
        recall_0 = set_metrics.recalls[0]
        f1_0 = set_metrics.f1_scores[0]

        return [model, iteration, set_name,
                acc,
                prec_1, recall_1, f1_1,
                prec_0, recall_0, f1_0]
