import copy
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from scripts.definitions.results import FitResults, FitMetrics
from scripts.prepare_data_for_training.samples.samples_v2 import SamplesDatasetV2
from scripts.utils.common import zero_division_safe


class FitInput:
    def __init__(self,
                 nn,
                 optim,
                 train_loss,
                 dev_loss,
                 test_loss,
                 train_ds: SamplesDatasetV2,
                 valid_ds: SamplesDatasetV2,
                 train_loader: torch.utils.data.DataLoader,
                 valid_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader,
                 epochs,
                 labels_size):
        self.nn = nn
        self.optim = optim

        self.train_loss = train_loss
        self.dev_loss = dev_loss
        self.test_loss = test_loss

        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.epochs = epochs
        self.labels_size = labels_size


class FitOutput:
    def __init__(self,
                 best_model,
                 fit_results: FitResults):
        self.best_model = best_model
        self.fit_results = fit_results


class Fitter:

    def __init__(self, fit_input: FitInput):
        self.fit_results = FitResults()

        self.nn = fit_input.nn
        self.opt = fit_input.optim

        self.train_loss = fit_input.train_loss
        self.dev_loss = fit_input.dev_loss
        self.test_loss = fit_input.test_loss

        self.train_ds = fit_input.train_ds
        self.valid_ds = fit_input.valid_ds

        self.train_loader = fit_input.train_loader
        self.valid_loader = fit_input.valid_loader
        self.test_loader = fit_input.test_loader

        self.epochs = fit_input.epochs
        self.batches_per_epoch = len(self.train_loader)
        self.labels_size = fit_input.labels_size

    def fit(self):
        batches = []
        # Results
        fit_results: FitResults = FitResults()

        best_valid_loss = float('inf')
        best_model = copy.deepcopy(self.nn.state_dict())

        initial_validation_loss = self.get_dev_loss()
        epoch_validation_loss = 0

        pbar = tqdm(range(self.epochs))
        for epoch in pbar:

            # Reset epoch_loss
            cumulative_epoch_batches_mean_loss = 0
            epoch_batches_counter = 0

            # Init to 0 for the 1st run
            epoch_loss = 0

            for X, y, frame_names in self.train_loader:
                # batch_str = "".join([str(item) for item in y.tolist()])
                # batches.append(batch_str)

                """
                Train
                """
                self.nn.train()

                # Cuda
                torch.cuda.empty_cache()

                # Make prediction
                pred = self.nn(X.to('cuda'))

                # Calculate loss
                batch_loss_mean = self.train_loss(pred, y.to('cuda'))

                # Backpropagation
                batch_loss_mean.backward(retain_graph=True)

                # Opt step
                self.opt.step()

                # Zero gradient
                self.opt.zero_grad()

                """
                Save train mode results
                """
                # Update & save epoch_loss
                batch_loss_mean = batch_loss_mean.detach().item()
                fit_results.save_batch_loss(batch_loss_mean)

                cumulative_epoch_batches_mean_loss += batch_loss_mean

                # Calculate epoch loss
                # NOTE: validation loss only after whole epoch
                epoch_loss = cumulative_epoch_batches_mean_loss / (epoch_batches_counter + 1)

                # Print Losses
                progress = round(100 * epoch_batches_counter / self.batches_per_epoch, 1)

                # validation loss is always 1 epoch late
                print_val_loss = epoch_validation_loss if epoch > 0 else initial_validation_loss
                pbar.set_description(f'EL: {epoch_loss}, VL: {print_val_loss}, Progress: {progress}%')

                # Update batches counter
                epoch_batches_counter += 1

                """
                Eval
                """
                self.nn.eval()
                with torch.no_grad():
                    pred = self.nn(X.to('cuda'))

                    # Save metrics
                    batch_metrics = self.create_fit_metrics(pred.to('cpu'), y, frame_names)
                    fit_results.save_batch_metrics(batch_metrics)

            # Validation Dataset
            epoch_validation_loss = self.get_dev_loss()

            # Saving Best Result:
            if epoch_validation_loss < best_valid_loss:
                best_model = self.copy_nn()
                best_valid_loss = epoch_validation_loss

            # Save for epoch
            fit_results.save_epoch_loses(epoch_loss, epoch_validation_loss)

        self.nn.load_state_dict(best_model)

        fit_results.train_metrics = self.create_set_metrics(self.train_loader)
        print('Train done!')

        fit_results.dev_metrics = self.create_set_metrics(self.valid_loader)
        print('Dev done!')

        fit_results.test_metrics = self.create_set_metrics(self.test_loader)
        print('Test done!')

        # print("\n".join(batches))

        return FitOutput(best_model, fit_results)

    def get_dev_loss(self):
        self.nn.eval()
        with torch.no_grad():

            epoch_validation_loss = 0
            for X, y, frame_names in self.valid_loader:
                epoch_validation_loss += self.dev_loss(self.nn(X.to('cuda')), y.to('cuda')).detach().item()

            # multiply by train/valid ratio? bez sensu
            # epoch_validation_loss *= (len(self.train_ds) / len(self.valid_ds))

            epoch_validation_loss /= len(self.valid_loader)

            return epoch_validation_loss

    def copy_nn(self):
        return copy.deepcopy(self.nn.state_dict())

    def create_set_metrics(self, set_loader):
        fit_metrics: List[FitMetrics] = []

        for X, y, frame_names in set_loader:
            prediction = self.nn(X.to('cuda')).to('cpu')
            metrics = self.create_fit_metrics(prediction, y, frame_names)
            fit_metrics.append(metrics)

        return FitMetrics.from_multiple(fit_metrics)

    @staticmethod
    def create_fit_metrics(pred, y, frame_names):
        frame_names = np.asarray(frame_names)

        """
        pred shape - [batch_size,2]
        y shape - [batch_size]
        """
        # for every [score for 0, score for 1] get index of max score (not score itself) => get winning label
        pred = pred.max(1)[1]

        # all with shape [batch_size]
        actual_0 = y == 0
        actual_1 = y == 1
        pred_0 = pred == 0
        pred_1 = pred == 1

        """
        https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9
                Predictions
        Actual  0   1
        0       00  01
        1       10  11
        """

        m_00 = actual_0.logical_and(actual_0 == pred_0).numpy()
        m_01 = actual_0.logical_and(actual_0 == pred_1).numpy()
        m_10 = actual_1.logical_and(actual_1 == pred_0).numpy()
        m_11 = actual_1.logical_and(actual_1 == pred_1).numpy()

        f_00 = frame_names[m_00]
        f_01 = frame_names[m_01]
        f_10 = frame_names[m_10]
        f_11 = frame_names[m_11]

        c_00 = len(f_00)
        c_01 = len(f_01)
        c_10 = len(f_10)
        c_11 = len(f_11)

        acc = (pred == y).float().mean().item()

        precision0 = zero_division_safe(lambda _: c_00 / (c_00 + c_10))
        precision1 = zero_division_safe(lambda _: c_11 / (c_11 + c_01))
        precisions = np.array([precision0, precision1])

        recall0 = zero_division_safe(lambda _: c_00 / (c_00 + c_01))
        recall1 = zero_division_safe(lambda _: c_11 / (c_10 + c_11))
        recalls = np.array([recall0, recall1])

        up = 2 * (precisions * recalls)
        down = precisions + recalls
        f1_scores = np.divide(up, down, out=np.zeros_like(up), where=up != 0)

        return FitMetrics(acc,
                          precisions.tolist(), recalls.tolist(), f1_scores.tolist(),
                          f_00.tolist(), f_01.tolist(), f_10.tolist(), f_11.tolist())
