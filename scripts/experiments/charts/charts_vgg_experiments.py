import time
from datetime import datetime

from scripts.definitions.params import RunParams
from scripts.training.tests_runner import TestRunner


def add_date(value):
    date_now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return f"{value}_{date_now}"


def run_experiments_with_split_per_subject(name, model, t_model, optimizer):
    run_params = RunParams.from_json_dict({
        "name": name,
        "output_dir": "charts/model_experiments_subject",
        "dataset_params": {
            "samples_file_name": "chart_images_5s_pure_fix",
            "samples_split": "chart_images_5s_pure_by_subject",
            "labels_weights": [1, 1],
        },
        "train_params": {
            "batch_size": 128,
            "epochs": 100,
        },
        "iterations": 10,
        "models": [model],
        "t_models": [t_model],
        "optimizer": optimizer
    })
    run_results = TestRunner(run_params).run()


def run_experiments_with_split_per_video(name, model, t_model, optimizer):
    run_params = RunParams.from_json_dict({
        "name": name,
        "output_dir": "charts/model_experiments_video",
        "dataset_params": {
            "samples_file_name": "chart_images_5s_pure_fix",
            "samples_split": "chart_images_5s_pure_by_video",
            "labels_weights": [1, 1],
        },
        "train_params": {
            "batch_size": 128,
            "epochs": 100,
        },
        "iterations": 10,
        "models": [model],
        "t_models": [t_model],
        "optimizer": optimizer
    })
    run_results = TestRunner(run_params).run()


def charts_vgg_experiments():
    models = ["m_classic",
              "m_classic_drop1",
              "m_classic_drop_old",
              "m_classic_less_layers",
              "m_classic_less_nodes",
              "m_classic_xavier",
              "m_classic_less_layers_drop",
              "m_classic_small_both_drop",
              "m_classic_small_class_drop",
              "m_classic_small_feature_drop"]

    optimizers = ["adam_lr_1e3_wd_1e2",
                  "adam_lr_1e3_wd_5e2",
                  "adam_lr_1e3_wd_1e1",
                  "adam_lr_1e3_wd_5e1",
                  "adam_lr_1e4_wd_1e2",
                  "adam_lr_1e4_wd_5e2",
                  "adam_lr_1e4_wd_1e1",
                  "adam_lr_1e4_wd_5e1",
                  "adam_lr_1e5_wd_1e2",
                  "adam_lr_1e5_wd_5e2",
                  "adam_lr_1e5_wd_1e1",
                  "adam_lr_1e5_wd_5e1"]

    tmodels = ["vgg"]

    start = time.time()

    for model in models:
        for optimizer in optimizers:
            for tmodel in tmodels:
                name = f"test_{tmodel}_{model}_{optimizer}"
                run_experiments_with_split_per_subject(name, model, tmodel, optimizer)
                run_experiments_with_split_per_video(name, model, tmodel, optimizer)

    end = time.time()
    print(end - start)


if __name__ == "__main__":
    """
    Run all VGG experiments for charts
    """
    charts_vgg_experiments()
