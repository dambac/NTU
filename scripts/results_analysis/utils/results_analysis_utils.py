import json

from scripts.definitions.results import RunResults
from scripts.utils.constants import C


def load_results_from_full_dir_path(path):
    results_path = f"{path}/results.json"
    with open(results_path, 'r') as file:
        results_dict = json.load(file)
        return RunResults.from_json_dict(results_dict)


def load_results(results_dir):
    results_path = f"{C.RESULTS_DIR}/{results_dir}/results.json"
    with open(results_path, 'r') as file:
        results_dict = json.load(file)
        return RunResults.from_json_dict(results_dict)


def load_results2(results_dir):
    results_path = f"{C.RESULTS_DIR}/{results_dir}/results2.json"
    with open(results_path, 'r') as file:
        results_dict = json.load(file)
        return RunResults.from_json_dict(results_dict)


def save_results2(results: RunResults, results_dir):
    results_path = f"{C.RESULTS_DIR}/{results_dir}/results2.json"
    with open(results_path, "w") as file:
        json.dump(results.to_json_dict(), file, indent=2)


def save_results3(results: RunResults, results_dir):
    results_path = f"{C.RESULTS_DIR}/{results_dir}/results3.json"
    with open(results_path, "w") as file:
        json.dump(results.to_json_dict(), file, indent=2)
