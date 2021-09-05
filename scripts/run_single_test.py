from datetime import datetime

from definitions.params import RunParams
from training.tests_runner import TestRunner


def add_date(value):
    date_now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return f"{value}_{date_now}"


def run():
    run_params = RunParams.from_json_dict({
        "name": add_date("samples_website_weights_1_50"),
        "dataset_params": {
            "samples_file_name": "websites_10s_pure",
            "samples_split": "websites_10s_pure_by_video",
            "labels_weights": [1, 50],
            # "labels_weights": [0.5, 0.6]
        },
        "train_params": {
            "batch_size": 128,
            "epochs": 5,
        },
        "iterations": 2,
        "models": ["classic"],
        # "t_models": ["alex"]
        "t_models": ["alex", "alex_deep", "resnet", "vgg", "vgg_deep"]
    })
    run_results = TestRunner(run_params).run()
    z = 1


def run_website(weights, samples_name, name):
    run_params = RunParams.from_json_dict({
        "name": name,
        "output_dir": "websites",
        "dataset_params": {
            "samples_file_name": samples_name,
            "samples_split": "websites_10s_pure_by_video",
            "labels_weights": weights,
            # "labels_weights": [0.5, 0.6]
        },
        "train_params": {
            "batch_size": 128,
            "epochs": 100,
        },
        "iterations": 10,
        "models": ["classic"],
        # "t_models": ["alex"]
        "t_models": ["alex", "alex_deep", "resnet", "vgg", "vgg_deep"]
    })
    run_results = TestRunner(run_params).run()
    z = 1


if __name__ == "__main__":
    run()
    # weights_dict = {
    #     "weight_none": None,
    #     "weights_1_1": [1, 1],
    #     "weights_1_50": [1, 50]
    # }
    # for weights_name, weights in weights_dict.items():
    #     run_website(samples_name="websites_10s",
    #                 weights=weights,
    #                 name=f"samples_websites_10s_{weights_name}")
    #
    #     run_website(samples_name="websites_10s_pure",
    #                 weights=weights,
    #                 name=f"samples_websites_10s_pure_{weights_name}")

