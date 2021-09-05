import time
from datetime import datetime

from scripts.definitions.params import RunParams
from scripts.training.tests_runner import TestRunner


def add_date(value):
    date_now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return f"{value}_{date_now}"


def run_transfer_models_experiments():
    run_params = RunParams.from_json_dict({
        "name": "test_transfers",
        "output_dir": "charts/transfer",
        "dataset_params": {
            "samples_file_name": "chart_images_5s_pure",
            "samples_split": "chart_images_5s_pure_by_video",
            "labels_weights": [1, 1],
        },
        "train_params": {
            "batch_size": 128,
            "epochs": 80,
        },
        "iterations": 10,
        "models": ["m_classic"],
        "t_models": [
            "alexm2",
            "alex",
            "resnet",
            "vggm2",
            "vgg"
        ],
        "optimizer": "adam_lr_1e3_wd_1e2"
    })
    run_results = TestRunner(run_params).run()


def charts_transfer_comparison():
    start = time.time()

    run_transfer_models_experiments()

    end = time.time()
    print(end - start)


if __name__ == "__main__":
    charts_transfer_comparison()
