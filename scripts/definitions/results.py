import itertools
from typing import List

import numpy as np

from scripts.definitions.params import RunParams, Combination
from scripts.utils.serialization import to_json_dict, from_json_dict, list_to_str, list_from_str


class RunResults:
    """
    Results of a
    """

    def __init__(self,
                 run_params,
                 combination_results):
        self.run_params: RunParams = run_params
        self.combination_results: List[CombinationResults] = combination_results

    def get_combination_results(self, model, t_model):
        comb = Combination(model, t_model)
        return [cr for cr in self.combination_results if cr.combination.equals(comb)][0]

    def to_json_dict(self):
        return to_json_dict(self, {
            "run_params": self.run_params.to_json_dict(),
            "combination_results": [cr.to_json_dict() for cr in self.combination_results]
        })

    @staticmethod
    def from_json_dict(dic):
        return from_json_dict(dic, RunResults, {
            "run_params": lambda val: RunParams.from_json_dict(val),
            "combination_results": lambda val: [CombinationResults.from_json_dict(cr) for cr in val]
        })

    @staticmethod
    def empty():
        return RunResults(None, None)


class DatasetResults:
    """
    Contains details about samples and sets used in training
    """

    def __init__(self, labels_count, labels_distribution,
                 train_labels, train_labels_count, train_labels_distribution,
                 valid_labels, valid_labels_count, valid_labels_distribution,
                 test_labels, test_labels_count, test_labels_distribution):
        """

        :param labels_count: total number of samples used in learning
        :param labels_distribution: total number of samples for labels 0 and 1

        :param train_labels: names of all samples that ended up in train set
        :param train_labels_count: number of all samples in train set
        :param train_labels_distribution: number of train set samples for labels 0 and 1

        :param valid_labels: names of all samples that ended up in dev set
        :param valid_labels_count: number of all samples in dev set
        :param valid_labels_distribution: number of dev set samples for labels 0 and 1

        :param test_labels: names of all samples that ended up in test set
        :param test_labels_count: number of all samples in test set
        :param test_labels_distribution: number of test set samples for labels 0 and 1
        """
        self.labels_count = labels_count
        self.labels_distribution = labels_distribution

        self.train_labels = train_labels
        self.train_labels_count = train_labels_count
        self.train_labels_distribution = train_labels_distribution

        self.valid_labels = valid_labels
        self.valid_labels_count = valid_labels_count
        self.valid_labels_distribution = valid_labels_distribution

        self.test_labels = test_labels
        self.test_labels_count = test_labels_count
        self.test_labels_distribution = test_labels_distribution

    def to_json_dict(self):
        return to_json_dict(self, {
            "train_labels": list_to_str(self.train_labels),
            "valid_labels": list_to_str(self.valid_labels),
            "test_labels": list_to_str(self.test_labels),
        })

    @staticmethod
    def from_json_dict(dic):
        return from_json_dict(dic, DatasetResults, {
            "train_labels": lambda val: list_from_str(val, string_values=True),
            "valid_labels": lambda val: list_from_str(val, string_values=True),
            "test_labels": lambda val: list_from_str(val, string_values=True)
        })

    @staticmethod
    def empty():
        return DatasetResults(None, None,
                              None, None, None,
                              None, None, None,
                              None, None, None)


class CombinationResults:
    """
    Results for a single combination only
    """

    def __init__(self, combination, dataset_results, iterations_results):
        """

        :param combination: name of combination
        :param dataset_results: details about dataset
        :param iterations_results: results achieved in each iteration
        """
        self.combination: Combination = combination
        self.dataset_results = dataset_results
        self.iterations_results: List[IterationResults] = iterations_results

    def to_json_dict(self):
        return to_json_dict(self, {
            "combination": self.combination.to_json_dict(),
            "dataset_results": self.dataset_results.to_json_dict(),
            "iterations_results": [ir.to_json_dict() for ir in self.iterations_results]
        })

    @staticmethod
    def from_json_dict(dic):
        return from_json_dict(dic, CombinationResults, {
            "combination": lambda val: Combination.from_json_dict(val),
            "dataset_results": lambda val: DatasetResults.from_json_dict(val),
            "iterations_results": lambda val: [IterationResults.from_json_dict(ir) for ir in val]
        })

    @staticmethod
    def empty():
        return CombinationResults(None, None, None)


class IterationResults:
    """
    Results achieved in a single iteration of training.
    NOTE: one iteration does not mean one epoch - iteration means a whole training, which consists of multiple epochs.
    """

    def __init__(self, iteration, fit_results, best_model_file, execution_time):
        # number of iteration starting from 0
        self.iteration = iteration
        # training results
        self.fit_results: FitResults = fit_results
        # path to a file with serialized best neural network model
        self.best_model_file = best_model_file

        self.execution_time = execution_time

    def to_json_dict(self):
        return to_json_dict(self, {
            "fit_results": self.fit_results.to_json_dict()
        })

    @staticmethod
    def from_json_dict(dic):
        return from_json_dict(dic, IterationResults, {
            "fit_results": lambda val: FitResults.from_json_dict(val),
        })

    @staticmethod
    def empty():
        return IterationResults(None, None, None, None)


class FitResults:
    """
    Results of neural network training
    """

    def __init__(self):
        # metrics for each batch
        self.batch_metrics: List[FitMetrics] = []
        # train set losses for each batch
        self.batch_train_loses = []

        # train set losses for each epoch
        self.epoch_train_loses = []
        # dev set losses for each epoch
        self.epoch_dev_loses = []

        self.train_metrics: FitMetrics = None
        self.dev_metrics: FitMetrics = None
        self.test_metrics: FitMetrics = None

    def save_batch_loss(self, batch_loss):
        self.batch_train_loses.append(batch_loss)

    def save_epoch_loses(self, train_loss, dev_loss):
        self.epoch_train_loses.append(train_loss)
        self.epoch_dev_loses.append(dev_loss)

    def save_batch_metrics(self, fit_metrics):
        self.batch_metrics.append(fit_metrics)

    def to_json_dict(self):
        return to_json_dict(self, {
            "batch_metrics": [metric.to_json_dict() for metric in self.batch_metrics],
            "batch_train_loses": list_to_str(self.batch_train_loses),
            "epoch_train_loses": list_to_str(self.epoch_train_loses),
            "epoch_dev_loses": list_to_str(self.epoch_dev_loses),
            "train_metrics": self.train_metrics.to_json_dict(),
            "dev_metrics": self.dev_metrics.to_json_dict(),
            "test_metrics": self.test_metrics.to_json_dict()
        })

    @staticmethod
    def from_json_dict(dic):
        return from_json_dict(dic, FitResults, {
            "batch_metrics": lambda val: [FitMetrics.from_json_dict(item) for item in val],
            "batch_train_loses": lambda val: list_from_str(val),
            "epoch_train_loses": lambda val: list_from_str(val),
            "epoch_dev_loses": lambda val: list_from_str(val),
            "train_metrics": lambda val: FitMetrics.from_json_dict(val),
            "dev_metrics": lambda val: FitMetrics.from_json_dict(val),
            "test_metrics": lambda val: FitMetrics.from_json_dict(val)
        })

    @staticmethod
    def empty():
        return FitResults()


class FitMetrics:
    """
    Metrics describing neural network performance
    """

    def __init__(self, acc,
                 precisions, recalls, f1_scores,
                 frames00, frames01, frames10, frames11):
        """

        :param acc: accuracy
        :param precisions: array [1,2] which precision for label 0 and 1
        :param recalls: array [1,2] which recall for label 0 and 1
        :param f1_scores: array [1,2] which F1 score for label 0 and 1
        :param frames00: names of all frames that were True Negatives
        :param frames01: names of all frames that were False Positives
        :param frames10: names of all frames that were False Negatives
        :param frames11: names of all frames that were True Positives
        """
        self.acc = acc

        # python lists
        self.precisions = precisions
        self.recalls = recalls
        self.f1_scores = f1_scores
        self.frames00 = frames00
        self.frames01 = frames01
        self.frames10 = frames10
        self.frames11 = frames11

    @staticmethod
    def from_multiple(multiple_metrics):
        multiple_metrics: List[FitMetrics]

        acc = np.array([metric.acc for metric in multiple_metrics]).mean()

        precisions = np.array([metric.precisions for metric in multiple_metrics]).mean(axis=0)
        recalls = np.array([metric.recalls for metric in multiple_metrics]).mean(axis=0)

        up = 2 * (precisions * recalls)
        down = precisions + recalls
        f1_scores = np.divide(up, down, out=np.zeros_like(up), where=up != 0)

        frames00 = list(itertools.chain(*[metric.frames00 for metric in multiple_metrics]))
        frames01 = list(itertools.chain(*[metric.frames01 for metric in multiple_metrics]))
        frames10 = list(itertools.chain(*[metric.frames10 for metric in multiple_metrics]))
        frames11 = list(itertools.chain(*[metric.frames11 for metric in multiple_metrics]))

        return FitMetrics(acc,
                          precisions.tolist(), recalls.tolist(), f1_scores.tolist(),
                          frames00, frames01, frames10, frames11)

    def to_json_dict(self):
        return to_json_dict(self, {
            "precisions": list_to_str(self.precisions),
            "frames00": list_to_str(self.frames00),
            "frames01": list_to_str(self.frames01),
            "frames10": list_to_str(self.frames10),
            "frames11": list_to_str(self.frames11)
        })

    @staticmethod
    def from_json_dict(dic):
        return from_json_dict(dic, FitMetrics, {
            "precisions": lambda val: list_from_str(val),
            "frames00": lambda val: list_from_str(val, string_values=True),
            "frames01": lambda val: list_from_str(val, string_values=True),
            "frames10": lambda val: list_from_str(val, string_values=True),
            "frames11": lambda val: list_from_str(val, string_values=True)
        })

    @staticmethod
    def empty():
        return FitMetrics(None,
                          None, None, None,
                          None, None, None, None)
