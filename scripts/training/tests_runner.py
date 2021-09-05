import json

from scripts.definitions.models.model_creator import create_model
from scripts.definitions.params import RunParams, get_t_model_params, Combination
from scripts.definitions.results import DatasetResults, IterationResults, RunResults, CombinationResults
from scripts.definitions.models.optimizers import create_optimizer
from scripts.prepare_data_for_training.samples.convert_to_full_samples import samples_to_full_samples
from scripts.training.fitter import FitInput, Fitter, FitOutput
from scripts.prepare_data_for_training.samples.samples_v2 import *
from scripts.utils.constants import C
from scripts.utils.serialization import read_json
import time


class FullSamplesCache:
    def __init__(self):
        self.samples_by_t_model = {}

    def has_for_t_model(self, t_model_name):
        return t_model_name in self.samples_by_t_model.keys()

    def get_for_t_model(self, t_model_name) -> List[FullSampleV2]:
        return self.samples_by_t_model[t_model_name]

    def store_for_t_model(self, t_model_name, samples_and_ids):
        self.samples_by_t_model[t_model_name] = samples_and_ids


class TestRunner:
    def __init__(self, run_params: RunParams):
        self.run_params = run_params
        self.train_params = run_params.train_params
        self.dataset_params = run_params.dataset_params

        self.samples = self._get_samples()
        labels = [s.label for s in self.samples]
        self.labels_size = len(set(labels))

        self.full_samples_cache = FullSamplesCache()

        self.output_dir = f"{C.RESULTS_DIR}/{run_params.output_dir}/{run_params.name}"
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)

        self.iterations = run_params.iterations
        self.combinations = self._prepare_all_combinations()

    def run(self):
        print(f"Running: {self.run_params.name}")
        combinations_results = []
        for c in self.combinations:
            combinations_results.append(self._run_combination(c))

        results = RunResults(self.run_params, combinations_results)

        results.to_json_dict()
        with open(f"{self.output_dir}/results.json", "w") as file:
            json.dump(results.to_json_dict(), file, indent=2)
        return results

    def _get_samples(self):
        if self.dataset_params.samples_file_name:
            samples = SampleV2.from_csv(self.dataset_params.samples_file_name)
        else:
            raise Exception("No samples file")

        # if self.dataset_params.samples_limit:
        #     samples = get_n_random(samples, self.dataset_params.samples_limit)

        return samples

    def _run_combination(self, combination):
        """
        Create output dir for combination
        """
        combination_dir = f"{self.output_dir}/{combination.model}-{combination.t_model}"
        Path(combination_dir).mkdir(exist_ok=True)

        """
        Create samples for current t_model
        """
        model = combination.model
        t_model_params = get_t_model_params(combination.t_model)

        samples: List[FullSampleV2]
        samples, samples_ids = self._get_full_samples_for_combination(combination)

        # if self.dataset_params.sets_absolutes:
        #     ids_subsets = split_into_sets_absolute(samples_ids, self.dataset_params.sets_absolutes)
        # else:
        #     ids_subsets = split_into_sets(samples_ids, self.dataset_params.sets_ratios)
        ids_subsets = self._get_samples_subsets_ids()
        z = 1

        """
        Save labels distributions
        """
        dataset_results = self._prepare_dataset_results(samples, ids_subsets)

        """
        Run iterations
        """
        iterations_results = []
        for it in range(self.iterations):
            print(f"Running iteration: {it}/{self.iterations - 1}")

            """
            Setup
            """
            epochs = self.train_params.epochs
            batch_size = self.train_params.batch_size

            train_ds = SamplesDatasetV2(samples, ids_subsets[0])
            valid_ds = SamplesDatasetV2(samples, ids_subsets[1])
            test_ds = SamplesDatasetV2(samples, ids_subsets[2])
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

            nn = create_model(model, t_model_params.output_size)
            opt = create_optimizer(self.run_params.optimizer, nn)

            if self.dataset_params.labels_weights:
                weights = torch.FloatTensor(self.dataset_params.labels_weights).to('cuda')
                loss = torch.nn.CrossEntropyLoss(weights)
            else:
                label_0_size = len([s for s in samples if s.label == 0])
                label_1_size = len([s for s in samples if s.label == 1])
                max_label_size = max(label_0_size, label_1_size)

                labels_weights = [max_label_size / label_0_size,
                                  max_label_size / label_1_size]
                weights = torch.FloatTensor(labels_weights).to('cuda')
                loss = torch.nn.CrossEntropyLoss(weights)

            fit_input = FitInput(
                nn=nn,
                optim=opt,
                train_loss=loss,
                dev_loss=loss,
                test_loss=loss,
                train_ds=train_ds,
                valid_ds=valid_ds,
                train_loader=train_loader,
                valid_loader=valid_loader,
                test_loader=test_loader,
                epochs=epochs,
                labels_size=self.labels_size
            )

            """
            Fit nn
            """
            start = time.time()
            fit_output: FitOutput = Fitter(fit_input).fit()
            stop = time.time()
            execution_time = stop - start

            """
            Save best model
            """
            best_model = fit_output.best_model
            best_model_file = f"{combination_dir}/best_model_{it}.pt"
            torch.save(best_model, best_model_file)

            """
            Save iterations results
            """
            fit_results = fit_output.fit_results
            iteration_results = IterationResults(it, fit_results, best_model_file, execution_time)
            iterations_results.append(iteration_results)

        return CombinationResults(combination, dataset_results, iterations_results)

    def _get_samples_subsets_ids(self):
        samples_split_name = self.dataset_params.samples_split
        samples_split = read_json(f"{C.DistributionsAndSets.SETS_SPLITS_SAMPLES}/{samples_split_name}.json")

        subsets = []
        for set_name, sample_ids in samples_split.items():
            subsets.append(sample_ids)
        return subsets

    def _get_full_samples_for_combination(self, combination):
        t_model_name = combination.t_model
        t_model_params = get_t_model_params(t_model_name)

        if self.full_samples_cache.has_for_t_model(t_model_name):
            samples, samples_ids = self.full_samples_cache.get_for_t_model(t_model_name)
        else:
            samples, samples_ids = samples_to_full_samples(self.samples, t_model_params.views_path)
            self.full_samples_cache.store_for_t_model(t_model_name, (samples, samples_ids))

        return samples, samples_ids

    def _prepare_dataset_results(self, samples: List[FullSampleV2], ids_subsets) -> DatasetResults:
        labels_count = len(samples)
        labels_distribution = self._count_number_of_samples_per_label(samples)

        train_labels = ids_subsets[0]
        train_labels_count, train_labels_distribution = self._get_subset_count_and_dist(samples, train_labels)

        valid_labels = ids_subsets[1]
        valid_labels_count, valid_labels_distribution = self._get_subset_count_and_dist(samples, valid_labels)

        test_labels = ids_subsets[2]
        test_labels_count, test_labels_distribution = self._get_subset_count_and_dist(samples, test_labels)

        return DatasetResults(
            labels_count, labels_distribution,
            train_labels, train_labels_count, train_labels_distribution,
            valid_labels, valid_labels_count, valid_labels_distribution,
            test_labels, test_labels_count, test_labels_distribution
        )

    def _prepare_all_combinations(self):
        combinations = []
        for model in self.run_params.models:
            for t_model in self.run_params.t_models:
                combinations.append(Combination(model, t_model))
        return combinations

    def _get_subset_count_and_dist(self, samples: List[FullSampleV2], subset_ids):
        subset_samples = [s for s in samples if s.id in subset_ids]
        return len(subset_samples), self._count_number_of_samples_per_label(subset_samples)

    def _count_number_of_samples_per_label(self, samples: List[FullSampleV2]):
        labels_dict = {}
        for s in samples:
            label = s.label.item()
            if label not in labels_dict:
                labels_dict[label] = 0
            labels_dict[label] = labels_dict[label] + 1
        return labels_dict
