import re

import pandas as pd

from scripts.results_post_processing.util import merge_dfs


def get_wd(experiment_name):
    # test_vgg_m_classic_adam_lr_1e3_wd_5e1 -> wd_5e1
    return re.search("(.*)_(wd_.*)", experiment_name).group(2)


def get_model_name(experiment_name):
    # test_vgg_m_classic_adam_lr_1e3_wd_5e1 -> m_classic_adam
    return re.search("(.*)test_vgg_m_(.+)_lr_(.*)", experiment_name).group(2)


def get_lr(experiment_name):
    # test_vgg_m_classic_adam_lr_1e3_wd_5e1 -> lr_1e3
    return re.search("(.*)test_vgg_m_(.+)_(lr_.*)_wd.*", experiment_name).group(3)


def get_split(experiment_name):
    # websites/big/test_vgg_m_classic_adam_lr_1e3_wd_5e1 -> lr_1e3 -> video
    # dir_name = re.search("(.*)/(.*)/test_vgg.*", experiment_name).group(2)
    if "subject" in experiment_name:
        return "subject"
    return "video"


class ResultsSummary:

    def __init__(self,
                 experiment_name,
                 iteration_results):
        self.experiment_name = experiment_name
        self.iteration_results = iteration_results

    @staticmethod
    def load(file, experiment_name):
        df = pd.read_csv(file)

        number_of_iterations = df["Iteration"].unique()
        iteration_results = []

        for iteration in number_of_iterations:
            iteration_df = df[df["Iteration"] == iteration]

            train_row = iteration_df[iteration_df["Set"] == "Train"].iloc[0]
            train_results = SetResults.from_df(train_row)

            dev_row = iteration_df[iteration_df["Set"] == "Dev"].iloc[0]
            dev_results = SetResults.from_df(dev_row)

            test_row = iteration_df[iteration_df["Set"] == "Test"].iloc[0]
            test_results = SetResults.from_df(test_row)

            iteration_results.append(
                IterationResults(
                    experiment_name=experiment_name,
                    iteration=iteration,
                    train_results=train_results,
                    dev_results=dev_results,
                    test_results=test_results
                )
            )

        return ResultsSummary(
            experiment_name=experiment_name,
            iteration_results=iteration_results
        )

    def __repr__(self):
        return self.experiment_name


class IterationResults:

    def __init__(self,
                 experiment_name,
                 iteration,
                 train_results,
                 dev_results,
                 test_results):
        self.experiment_name = experiment_name
        self.iteration = iteration

        self.train_results: SetResults = train_results
        self.dev_results: SetResults = dev_results
        self.test_results: SetResults = test_results

    @staticmethod
    def to_df_multiple(iteration_results):
        dataframes = []
        for it_results in iteration_results:
            it_df = IterationResults.to_df(it_results)
            dataframes.append(it_df)

        df = merge_dfs(dataframes)
        return df

    def to_df(self):
        df_data = []

        for set_results in [self.train_results, self.dev_results, self.test_results]:
            set_name = set_results.set_name

            df_data.append([
                self.experiment_name,
                self.iteration,
                set_name,
                set_results.acc,
                set_results.prec1,
                set_results.rec1,
                set_results.f11,
                set_results.prec0,
                set_results.rec0,
                set_results.f10
            ])

        df = pd.DataFrame(data=df_data,
                          columns=[
                              'Experiment',
                              'Iteration',
                              'Set',
                              'Acc',
                              'Prec 1', 'Rec 1', 'F1 1',
                              'Prec 0', 'Rec 0', 'F1 0'
                          ])
        return df

    def __repr__(self):
        return f"E: {self.experiment_name}, I: {self.iteration}"


class SetResults:
    def __init__(self,
                 set_name,
                 acc,
                 prec0,
                 prec1,
                 rec0,
                 rec1,
                 f10,
                 f11):
        self.set_name = set_name
        self.acc = acc
        self.prec0 = prec0
        self.prec1 = prec1
        self.rec0 = rec0
        self.rec1 = rec1
        self.f10 = f10
        self.f11 = f11

    @staticmethod
    def from_df(df):
        return SetResults(
            set_name=df["Set"],
            acc=df["Acc"],
            prec1=df["Prec 1"],
            rec1=df["Rec 1"],
            f11=df["F1 1"],
            prec0=df["Prec 0"],
            rec0=df["Rec 0"],
            f10=df["F1 0"],
        )
