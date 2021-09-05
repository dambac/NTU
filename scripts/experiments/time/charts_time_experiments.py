import time
from datetime import datetime

from scripts.definitions.params import RunParams
from scripts.training.tests_runner import TestRunner


def add_date(value):
    date_now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return f"{value}_{date_now}"


def run_time_experiment(name, model, t_model, optimizer):
    run_params = RunParams.from_json_dict({
        "name": name,
        "output_dir": "charts/time_experiments",
        "dataset_params": {
            "samples_file_name": "chart_images_5s_pure_fix",
            "samples_split": "chart_images_5s_pure_by_video",
            "labels_weights": [1, 1],
        },
        "train_params": {
            "batch_size": 128,
            "epochs": 5,
        },
        "iterations": 1,
        "models": [model],
        "t_models": [t_model],
        "optimizer": optimizer
    })
    run_results = TestRunner(run_params).run()


def charts_vgg_time_experiments():
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

    optimizers = ["adam_lr_1e3_wd_1e2"]

    tmodels = ["vgg"]

    start = time.time()

    for model in models:
        for optimizer in optimizers:
            for tmodel in tmodels:
                name = f"vgg_time_test_{tmodel}_{model}_{optimizer}"
                run_time_experiment(name, model, tmodel, optimizer)

    end = time.time()
    print(end - start)


def charts_transfer_models_time_experiments():
    models = ["m_classic"]

    optimizers = ["adam_lr_1e3_wd_1e2"]

    tmodels = ["alexm2", "alex", "resnet", "vgg", "vggm2"]

    start = time.time()

    for model in models:
        for optimizer in optimizers:
            for tmodel in tmodels:
                name = f"transfer_time_test_{tmodel}_{model}_{optimizer}"
                run_time_experiment(name, model, tmodel, optimizer)

    end = time.time()
    print(end - start)


if __name__ == "__main__":
    charts_vgg_time_experiments()
    charts_transfer_models_time_experiments()
