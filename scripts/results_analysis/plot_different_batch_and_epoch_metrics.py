from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from analysis.anal_utils import *
from scripts.definitions.results import *
from scripts.utils.common import mkdir_path


class Analyzer:
    def __init__(self, results_dir):
        self.results: RunResults = load_results(results_dir)
        self.base_output_dir = f"outputs/{results_dir}/metrics"
        Path(self.base_output_dir).mkdir(exist_ok=True, parents=True)

        self.run_params: RunParams = self.results.run_params
        self.epochs = self.run_params.train_params.epochs

        # intentional - analyze only first combination
        self.batches = len(self.results.combination_results[0].iterations_results[0].fit_results.batch_metrics)
        self.batches_per_epoch = int(self.batches / self.epochs)

    def analyze(self):
        combination_results: CombinationResults

        for combination_results in self.results.combination_results:
            self.analyze_for_combination(combination_results)

    def analyze_for_combination(self, combination_results):
        comb_name = f"{combination_results.combination.model}_{combination_results.combination.t_model}"
        output_dir = f"{self.base_output_dir}/{comb_name}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for iteration, iteration_results in enumerate(combination_results.iterations_results):
            self.analyze_for_iteration2(output_dir, iteration, iteration_results)

    def analyze_for_iteration2(self, output_dir, iteration, iteration_results: IterationResults):
        output_path = f"{output_dir}/{iteration}.jpg"

        fit_results: FitResults = iteration_results.fit_results

        batch_train_loss = self.get_batch_train_losses(fit_results)
        epoch_train_loss = self.get_epoch_train_losses(fit_results)
        epoch_dev_loss = self.get_epoch_dev_losses(fit_results)

        sns.reset_orig()  # get default matplotlib styles back
        colors = sns.color_palette('husl', n_colors=6)  # a list of RGB tuples
        # colors = _get_colors(6)

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

        xs = range(1, self.batches + 1)

        # first - losses
        ax0 = axs
        # ax0.set_ylim(ymin=0)
        ax0.plot(xs, batch_train_loss, color=colors[0], label="Train batch loss")
        ax0.plot(xs, epoch_train_loss, color=colors[1], label="Train epoch loss")
        ax0.plot(xs, epoch_dev_loss, color=colors[2], label="Dev epoch loss")
        self.add_legend(ax0)

        mkdir_path(output_path)
        fig.savefig(output_path, pad_inches=0.1, bbox_inches='tight')
        fig.clf()

    def analyze_for_iteration(self, output_dir, iteration, iteration_results: IterationResults):
        output_path = f"{output_dir}/{iteration}.jpg"

        fit_results: FitResults = iteration_results.fit_results

        batch_acc = self.get_batch_accs(fit_results)
        batch_precisions0 = self.get_batch_precisions(fit_results, 0)
        batch_precisions1 = self.get_batch_precisions(fit_results, 1)
        batch_train_loss = self.get_batch_train_losses(fit_results)
        epoch_train_loss = self.get_epoch_train_losses(fit_results)
        epoch_dev_loss = self.get_epoch_dev_losses(fit_results)

        sns.reset_orig()  # get default matplotlib styles back
        colors = sns.color_palette('husl', n_colors=6)  # a list of RGB tuples
        # colors = _get_colors(6)

        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 15), gridspec_kw={'height_ratios': [2, 1]})

        xs = range(1, self.batches + 1)

        # first - losses
        ax0 = axs[0]
        # ax0.set_ylim(ymin=0)
        ax0.plot(xs, batch_train_loss, color=colors[0], label="Train batch loss")
        ax0.plot(xs, epoch_train_loss, color=colors[1], label="Train epoch loss")
        ax0.plot(xs, epoch_dev_loss, color=colors[2], label="Dev epoch loss")
        self.add_legend(ax0)

        # second - metrics
        ax1 = axs[1]
        ax1.set_ylim([0, 1])
        ax1.plot(xs, batch_acc, color=colors[3], label="Train acc")
        ax1.plot(xs, batch_precisions0, '--', color=colors[4], label="Train prec0")
        ax1.plot(xs, batch_precisions1, '--', color=colors[5], label="Train prec1")
        self.add_legend(ax1)

        mkdir_path(output_path)
        fig.savefig(output_path, pad_inches=0.1, bbox_inches='tight')
        fig.clf()

    @staticmethod
    def add_legend(ax):
        handles, labels = ax.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    @staticmethod
    def get_batch_precisions(fit_results: FitResults, precision_index):
        precisions = []

        batch_metrics: FitMetrics
        for batch_metrics in fit_results.batch_metrics:
            precisions.append(batch_metrics.precisions[precision_index])

        return precisions

    @staticmethod
    def get_batch_accs(fit_results: FitResults):
        accs = []

        batch_metrics: FitMetrics
        for batch_metrics in fit_results.batch_metrics:
            accs.append(batch_metrics.acc)

        return accs

    def get_epoch_train_losses(self, fit_results: FitResults):
        losses = []

        for loss in fit_results.epoch_train_loses:
            losses.extend([loss] * self.batches_per_epoch)

        return losses

    def get_epoch_dev_losses(self, fit_results: FitResults):
        losses = []

        for loss in fit_results.epoch_dev_loses:
            losses.extend([loss] * self.batches_per_epoch)

        return losses

    def get_batch_train_losses(self, fit_results: FitResults):
        losses = []

        for loss in fit_results.batch_train_loses:
            losses.append(loss)

        return losses
