import unittest

from analysis import analyze_labels_in_subjects_dist, analyze_predicted_frames, verify_frame, \
    analyze_samples
from results_analysis.fix_results import correct_metrics
from results_analysis import analyze_results_basic, analyze_results_sorted_inside_combination, analyze_metrics
from analysis.statepoint import compare_frames
from dataset_analysis import visualize_state_events
from analysis.statepoint.analyze_events_for_video import StatePointAnalyzer
from definitions.domain.event import Event
from definitions.params import Combination
from utils import frames
from utils.constants import C


class AnalysisRunner(unittest.TestCase):

    def test_analyze_transfer_models(self):
        result_dirs = [
            "charts/big_duplicate/test_vgg_m_classic_adam_lr_1e3_wd_1e1",
            "charts/big_duplicate/test_vgg_m_classic_adam_lr_1e3_wd_1e2",
            "charts/big_duplicate/test_vgg_m_classic_adam_lr_1e3_wd_5e1",
            "charts/big_duplicate/test_vgg_m_classic_adam_lr_1e3_wd_5e2",
            "charts/big_duplicate/test_vgg_m_classic_adam_lr_1e5_wd_1e1",
            "charts/big_duplicate/test_vgg_m_classic_adam_lr_1e5_wd_1e2",
            "charts/big_duplicate/test_vgg_m_classic_adam_lr_1e5_wd_5e1",
            "charts/big_duplicate/test_vgg_m_classic_adam_lr_1e5_wd_5e2",
            "charts/big_duplicate/test_vgg_m_classic_drop1_adam_lr_1e3_wd_1e1",
            "charts/big_duplicate/test_vgg_m_classic_drop1_adam_lr_1e3_wd_1e2",
            "charts/big_duplicate/test_vgg_m_classic_drop1_adam_lr_1e3_wd_5e1",
            "charts/big_duplicate/test_vgg_m_classic_drop1_adam_lr_1e3_wd_5e2",
            "charts/big_duplicate/test_vgg_m_classic_drop1_adam_lr_1e5_wd_1e1",
            "charts/big_duplicate/test_vgg_m_classic_drop1_adam_lr_1e5_wd_1e2",
            "charts/big_duplicate/test_vgg_m_classic_drop1_adam_lr_1e5_wd_5e1",
            "charts/big_duplicate/test_vgg_m_classic_drop1_adam_lr_1e5_wd_5e2",
            "charts/big_duplicate/test_vgg_m_classic_drop_old_adam_lr_1e3_wd_1e1",
            "charts/big_duplicate/test_vgg_m_classic_drop_old_adam_lr_1e3_wd_1e2",
            "charts/big_duplicate/test_vgg_m_classic_drop_old_adam_lr_1e3_wd_5e1",
            "charts/big_duplicate/test_vgg_m_classic_drop_old_adam_lr_1e3_wd_5e2",
            "charts/big_duplicate/test_vgg_m_classic_drop_old_adam_lr_1e5_wd_1e1",
            "charts/big_duplicate/test_vgg_m_classic_drop_old_adam_lr_1e5_wd_1e2",
            "charts/big_duplicate/test_vgg_m_classic_drop_old_adam_lr_1e5_wd_5e1",
            "charts/big_duplicate/test_vgg_m_classic_drop_old_adam_lr_1e5_wd_5e2",
            "charts/big_duplicate/test_vgg_m_classic_less_layers_adam_lr_1e3_wd_1e1",
            "charts/big_duplicate/test_vgg_m_classic_less_layers_adam_lr_1e3_wd_1e2",
            "charts/big_duplicate/test_vgg_m_classic_less_layers_adam_lr_1e3_wd_5e1",
            "charts/big_duplicate/test_vgg_m_classic_less_layers_adam_lr_1e3_wd_5e2",
            "charts/big_duplicate/test_vgg_m_classic_less_layers_adam_lr_1e5_wd_1e1",
            "charts/big_duplicate/test_vgg_m_classic_less_layers_adam_lr_1e5_wd_1e2",
            "charts/big_duplicate/test_vgg_m_classic_less_layers_adam_lr_1e5_wd_5e1",
            "charts/big_duplicate/test_vgg_m_classic_less_layers_adam_lr_1e5_wd_5e2",
            "charts/big_duplicate/test_vgg_m_classic_less_nodes_adam_lr_1e3_wd_1e1",
            "charts/big_duplicate/test_vgg_m_classic_less_nodes_adam_lr_1e3_wd_1e2",
            "charts/big_duplicate/test_vgg_m_classic_less_nodes_adam_lr_1e3_wd_5e1",
            "charts/big_duplicate/test_vgg_m_classic_less_nodes_adam_lr_1e3_wd_5e2",
            "charts/big_duplicate/test_vgg_m_classic_less_nodes_adam_lr_1e5_wd_1e1",
            "charts/big_duplicate/test_vgg_m_classic_less_nodes_adam_lr_1e5_wd_1e2",
            "charts/big_duplicate/test_vgg_m_classic_less_nodes_adam_lr_1e5_wd_5e1",
            "charts/big_duplicate/test_vgg_m_classic_less_nodes_adam_lr_1e5_wd_5e2",
            "charts/big_duplicate/test_vgg_m_classic_xavier_adam_lr_1e3_wd_1e1",
            "charts/big_duplicate/test_vgg_m_classic_xavier_adam_lr_1e3_wd_1e2",
            "charts/big_duplicate/test_vgg_m_classic_xavier_adam_lr_1e3_wd_5e1",
            "charts/big_duplicate/test_vgg_m_classic_xavier_adam_lr_1e3_wd_5e2",
            "charts/big_duplicate/test_vgg_m_classic_xavier_adam_lr_1e5_wd_1e1",
            "charts/big_duplicate/test_vgg_m_classic_xavier_adam_lr_1e5_wd_1e2",
            "charts/big_duplicate/test_vgg_m_classic_xavier_adam_lr_1e5_wd_5e1",
            "charts/big_duplicate/test_vgg_m_classic_xavier_adam_lr_1e5_wd_5e2",

            "charts/bigsubject/test_vgg_m_classic_adam_lr_1e3_wd_1e1",
            "charts/bigsubject/test_vgg_m_classic_adam_lr_1e3_wd_1e2",
            "charts/bigsubject/test_vgg_m_classic_adam_lr_1e3_wd_5e1",
            "charts/bigsubject/test_vgg_m_classic_adam_lr_1e3_wd_5e2",
            "charts/bigsubject/test_vgg_m_classic_adam_lr_1e5_wd_1e1",
            "charts/bigsubject/test_vgg_m_classic_adam_lr_1e5_wd_1e2",
            "charts/bigsubject/test_vgg_m_classic_adam_lr_1e5_wd_5e1",
            "charts/bigsubject/test_vgg_m_classic_adam_lr_1e5_wd_5e2",
            "charts/bigsubject/test_vgg_m_classic_drop1_adam_lr_1e3_wd_1e1",
            "charts/bigsubject/test_vgg_m_classic_drop1_adam_lr_1e3_wd_1e2",
            "charts/bigsubject/test_vgg_m_classic_drop1_adam_lr_1e3_wd_5e1",
            "charts/bigsubject/test_vgg_m_classic_drop1_adam_lr_1e3_wd_5e2",
            "charts/bigsubject/test_vgg_m_classic_drop1_adam_lr_1e5_wd_1e1",
            "charts/bigsubject/test_vgg_m_classic_drop1_adam_lr_1e5_wd_1e2",
            "charts/bigsubject/test_vgg_m_classic_drop1_adam_lr_1e5_wd_5e1",
            "charts/bigsubject/test_vgg_m_classic_drop1_adam_lr_1e5_wd_5e2",
            "charts/bigsubject/test_vgg_m_classic_drop_old_adam_lr_1e3_wd_1e1",
            "charts/bigsubject/test_vgg_m_classic_drop_old_adam_lr_1e3_wd_1e2",
            "charts/bigsubject/test_vgg_m_classic_drop_old_adam_lr_1e3_wd_5e1",
            "charts/bigsubject/test_vgg_m_classic_drop_old_adam_lr_1e3_wd_5e2",
            "charts/bigsubject/test_vgg_m_classic_drop_old_adam_lr_1e5_wd_1e1",
            "charts/bigsubject/test_vgg_m_classic_drop_old_adam_lr_1e5_wd_1e2",
            "charts/bigsubject/test_vgg_m_classic_drop_old_adam_lr_1e5_wd_5e1",
            "charts/bigsubject/test_vgg_m_classic_drop_old_adam_lr_1e5_wd_5e2",
            "charts/bigsubject/test_vgg_m_classic_less_layers_adam_lr_1e3_wd_1e1",
            "charts/bigsubject/test_vgg_m_classic_less_layers_adam_lr_1e3_wd_1e2",
            "charts/bigsubject/test_vgg_m_classic_less_layers_adam_lr_1e3_wd_5e1",
            "charts/bigsubject/test_vgg_m_classic_less_layers_adam_lr_1e3_wd_5e2",
            "charts/bigsubject/test_vgg_m_classic_less_layers_adam_lr_1e5_wd_1e1",
            "charts/bigsubject/test_vgg_m_classic_less_layers_adam_lr_1e5_wd_1e2",
            "charts/bigsubject/test_vgg_m_classic_less_layers_adam_lr_1e5_wd_5e1",
            "charts/bigsubject/test_vgg_m_classic_less_layers_adam_lr_1e5_wd_5e2",
            "charts/bigsubject/test_vgg_m_classic_less_nodes_adam_lr_1e3_wd_1e1",
            "charts/bigsubject/test_vgg_m_classic_less_nodes_adam_lr_1e3_wd_1e2",
            "charts/bigsubject/test_vgg_m_classic_less_nodes_adam_lr_1e3_wd_5e1",
            "charts/bigsubject/test_vgg_m_classic_less_nodes_adam_lr_1e3_wd_5e2",
            "charts/bigsubject/test_vgg_m_classic_less_nodes_adam_lr_1e5_wd_1e1",
            "charts/bigsubject/test_vgg_m_classic_less_nodes_adam_lr_1e5_wd_1e2",
            "charts/bigsubject/test_vgg_m_classic_less_nodes_adam_lr_1e5_wd_5e1",
            "charts/bigsubject/test_vgg_m_classic_less_nodes_adam_lr_1e5_wd_5e2",
            "charts/bigsubject/test_vgg_m_classic_xavier_adam_lr_1e3_wd_1e1",
            "charts/bigsubject/test_vgg_m_classic_xavier_adam_lr_1e3_wd_1e2",
            "charts/bigsubject/test_vgg_m_classic_xavier_adam_lr_1e3_wd_5e1",
            "charts/bigsubject/test_vgg_m_classic_xavier_adam_lr_1e3_wd_5e2",
            "charts/bigsubject/test_vgg_m_classic_xavier_adam_lr_1e5_wd_1e1",
            "charts/bigsubject/test_vgg_m_classic_xavier_adam_lr_1e5_wd_1e2",
            "charts/bigsubject/test_vgg_m_classic_xavier_adam_lr_1e5_wd_5e1",
            "charts/bigsubject/test_vgg_m_classic_xavier_adam_lr_1e5_wd_5e2",

            "websites/big/test_vgg_m_classic_adam_lr_1e3_wd_1e1",
            "websites/big/test_vgg_m_classic_adam_lr_1e3_wd_1e2",
            "websites/big/test_vgg_m_classic_adam_lr_1e3_wd_5e1",
            "websites/big/test_vgg_m_classic_adam_lr_1e3_wd_5e2",
            "websites/big/test_vgg_m_classic_adam_lr_1e5_wd_1e1",
            "websites/big/test_vgg_m_classic_adam_lr_1e5_wd_1e2",
            "websites/big/test_vgg_m_classic_adam_lr_1e5_wd_5e1",
            "websites/big/test_vgg_m_classic_adam_lr_1e5_wd_5e2",
            "websites/big/test_vgg_m_classic_drop1_adam_lr_1e3_wd_1e1",
            "websites/big/test_vgg_m_classic_drop1_adam_lr_1e3_wd_1e2",
            "websites/big/test_vgg_m_classic_drop1_adam_lr_1e3_wd_5e1",
            "websites/big/test_vgg_m_classic_drop1_adam_lr_1e3_wd_5e2",
            "websites/big/test_vgg_m_classic_drop1_adam_lr_1e5_wd_1e1",
            "websites/big/test_vgg_m_classic_drop1_adam_lr_1e5_wd_1e2",
            "websites/big/test_vgg_m_classic_drop1_adam_lr_1e5_wd_5e1",
            "websites/big/test_vgg_m_classic_drop1_adam_lr_1e5_wd_5e2",
            "websites/big/test_vgg_m_classic_drop_old_adam_lr_1e3_wd_1e1",
            "websites/big/test_vgg_m_classic_drop_old_adam_lr_1e3_wd_1e2",
            "websites/big/test_vgg_m_classic_drop_old_adam_lr_1e3_wd_5e1",
            "websites/big/test_vgg_m_classic_drop_old_adam_lr_1e3_wd_5e2",
            "websites/big/test_vgg_m_classic_drop_old_adam_lr_1e5_wd_1e1",
            "websites/big/test_vgg_m_classic_drop_old_adam_lr_1e5_wd_1e2",
            "websites/big/test_vgg_m_classic_drop_old_adam_lr_1e5_wd_5e1",
            "websites/big/test_vgg_m_classic_drop_old_adam_lr_1e5_wd_5e2",
            "websites/big/test_vgg_m_classic_less_layers_adam_lr_1e3_wd_1e1",
            "websites/big/test_vgg_m_classic_less_layers_adam_lr_1e3_wd_1e2",
            "websites/big/test_vgg_m_classic_less_layers_adam_lr_1e3_wd_5e1",
            "websites/big/test_vgg_m_classic_less_layers_adam_lr_1e3_wd_5e2",
            "websites/big/test_vgg_m_classic_less_layers_adam_lr_1e5_wd_1e1",
            "websites/big/test_vgg_m_classic_less_layers_adam_lr_1e5_wd_1e2",
            "websites/big/test_vgg_m_classic_less_layers_adam_lr_1e5_wd_5e1",
            "websites/big/test_vgg_m_classic_less_layers_adam_lr_1e5_wd_5e2",
            "websites/big/test_vgg_m_classic_less_nodes_adam_lr_1e3_wd_1e1",
            "websites/big/test_vgg_m_classic_less_nodes_adam_lr_1e3_wd_1e2",
            "websites/big/test_vgg_m_classic_less_nodes_adam_lr_1e3_wd_5e1",
            "websites/big/test_vgg_m_classic_less_nodes_adam_lr_1e3_wd_5e2",
            "websites/big/test_vgg_m_classic_less_nodes_adam_lr_1e5_wd_1e1",
            "websites/big/test_vgg_m_classic_less_nodes_adam_lr_1e5_wd_1e2",
            "websites/big/test_vgg_m_classic_less_nodes_adam_lr_1e5_wd_5e1",
            "websites/big/test_vgg_m_classic_less_nodes_adam_lr_1e5_wd_5e2",
            "websites/big/test_vgg_m_classic_xavier_adam_lr_1e3_wd_1e1",
            "websites/big/test_vgg_m_classic_xavier_adam_lr_1e3_wd_1e2",
            "websites/big/test_vgg_m_classic_xavier_adam_lr_1e3_wd_5e1",
            "websites/big/test_vgg_m_classic_xavier_adam_lr_1e3_wd_5e2",
            "websites/big/test_vgg_m_classic_xavier_adam_lr_1e5_wd_1e1",
            "websites/big/test_vgg_m_classic_xavier_adam_lr_1e5_wd_1e2",
            "websites/big/test_vgg_m_classic_xavier_adam_lr_1e5_wd_5e1",
            "websites/big/test_vgg_m_classic_xavier_adam_lr_1e5_wd_5e2",

            "websites/bigsubject/test_vgg_m_classic_adam_lr_1e3_wd_1e1",
            "websites/bigsubject/test_vgg_m_classic_adam_lr_1e3_wd_1e2",
            "websites/bigsubject/test_vgg_m_classic_adam_lr_1e3_wd_5e1",
            "websites/bigsubject/test_vgg_m_classic_adam_lr_1e3_wd_5e2",
            "websites/bigsubject/test_vgg_m_classic_adam_lr_1e5_wd_1e1",
            "websites/bigsubject/test_vgg_m_classic_adam_lr_1e5_wd_1e2",
            "websites/bigsubject/test_vgg_m_classic_adam_lr_1e5_wd_5e1",
            "websites/bigsubject/test_vgg_m_classic_adam_lr_1e5_wd_5e2",
            "websites/bigsubject/test_vgg_m_classic_drop1_adam_lr_1e3_wd_1e1",
            "websites/bigsubject/test_vgg_m_classic_drop1_adam_lr_1e3_wd_1e2",
            "websites/bigsubject/test_vgg_m_classic_drop1_adam_lr_1e3_wd_5e1",
            "websites/bigsubject/test_vgg_m_classic_drop1_adam_lr_1e3_wd_5e2",
            "websites/bigsubject/test_vgg_m_classic_drop1_adam_lr_1e5_wd_1e1",
            "websites/bigsubject/test_vgg_m_classic_drop1_adam_lr_1e5_wd_1e2",
            "websites/bigsubject/test_vgg_m_classic_drop1_adam_lr_1e5_wd_5e1",
            "websites/bigsubject/test_vgg_m_classic_drop1_adam_lr_1e5_wd_5e2",
            "websites/bigsubject/test_vgg_m_classic_drop_old_adam_lr_1e3_wd_1e1",
            "websites/bigsubject/test_vgg_m_classic_drop_old_adam_lr_1e3_wd_1e2",
            "websites/bigsubject/test_vgg_m_classic_drop_old_adam_lr_1e3_wd_5e1",
            "websites/bigsubject/test_vgg_m_classic_drop_old_adam_lr_1e3_wd_5e2",
            "websites/bigsubject/test_vgg_m_classic_drop_old_adam_lr_1e5_wd_1e1",
            "websites/bigsubject/test_vgg_m_classic_drop_old_adam_lr_1e5_wd_1e2",
            "websites/bigsubject/test_vgg_m_classic_drop_old_adam_lr_1e5_wd_5e1",
            "websites/bigsubject/test_vgg_m_classic_drop_old_adam_lr_1e5_wd_5e2",
            "websites/bigsubject/test_vgg_m_classic_less_layers_adam_lr_1e3_wd_1e1",
            "websites/bigsubject/test_vgg_m_classic_less_layers_adam_lr_1e3_wd_1e2",
            "websites/bigsubject/test_vgg_m_classic_less_layers_adam_lr_1e3_wd_5e1",
            "websites/bigsubject/test_vgg_m_classic_less_layers_adam_lr_1e3_wd_5e2",
            "websites/bigsubject/test_vgg_m_classic_less_layers_adam_lr_1e5_wd_1e1",
            "websites/bigsubject/test_vgg_m_classic_less_layers_adam_lr_1e5_wd_1e2",
            "websites/bigsubject/test_vgg_m_classic_less_layers_adam_lr_1e5_wd_5e1",
            "websites/bigsubject/test_vgg_m_classic_less_layers_adam_lr_1e5_wd_5e2",
            "websites/bigsubject/test_vgg_m_classic_less_nodes_adam_lr_1e3_wd_1e1",
            "websites/bigsubject/test_vgg_m_classic_less_nodes_adam_lr_1e3_wd_1e2",
            "websites/bigsubject/test_vgg_m_classic_less_nodes_adam_lr_1e3_wd_5e1",
            "websites/bigsubject/test_vgg_m_classic_less_nodes_adam_lr_1e3_wd_5e2",
            "websites/bigsubject/test_vgg_m_classic_less_nodes_adam_lr_1e5_wd_1e1",
            "websites/bigsubject/test_vgg_m_classic_less_nodes_adam_lr_1e5_wd_1e2",
            "websites/bigsubject/test_vgg_m_classic_less_nodes_adam_lr_1e5_wd_5e1",
            "websites/bigsubject/test_vgg_m_classic_less_nodes_adam_lr_1e5_wd_5e2",
            "websites/bigsubject/test_vgg_m_classic_xavier_adam_lr_1e3_wd_1e1",
            "websites/bigsubject/test_vgg_m_classic_xavier_adam_lr_1e3_wd_1e2",
            "websites/bigsubject/test_vgg_m_classic_xavier_adam_lr_1e3_wd_5e1",
            "websites/bigsubject/test_vgg_m_classic_xavier_adam_lr_1e3_wd_5e2",
            "websites/bigsubject/test_vgg_m_classic_xavier_adam_lr_1e5_wd_1e1",
            "websites/bigsubject/test_vgg_m_classic_xavier_adam_lr_1e5_wd_1e2",
            "websites/bigsubject/test_vgg_m_classic_xavier_adam_lr_1e5_wd_5e1",
            "websites/bigsubject/test_vgg_m_classic_xavier_adam_lr_1e5_wd_5e2",

        ]
        result_dirs = [
            "websites/transfer/test_transfers",
            "charts/transfer/test_transfers",
        ]
        for result_dir in result_dirs:
            analyze_results_basic.Analyzer(results_dir=f"{result_dir}").analyze()

    def test_analyze_model_experiments(self):
        dirs_with_results = [
            "websites/"
        ]

    def test_analyze_results_sorted_inside_combination(self):
        # result_dirs = [
        #     "all_models_with_split_websites_10s_pure_by_subject_weights_weight_none",
        #     "all_models_with_split_websites_10s_pure_by_subject_weights_weights_1_1",
        #     "all_models_with_split_websites_10s_pure_by_subject_weights_weights_1_5",
        #     "all_models_with_split_websites_10s_pure_by_video_weights_weight_none",
        #     "all_models_with_split_websites_10s_pure_by_video_weights_weights_1_1",
        #     "all_models_with_split_websites_10s_pure_by_video_weights_weights_1_5",
        # ]
        result_dirs = [
            # "websites/transfer/test_transfers",
            "charts/transfer/test_transfers",

        ]
        for result_dir in result_dirs:
            analyze_results_sorted_inside_combination.Analyzer(results_dir=f"{result_dir}").analyze()

    def test_correct_results(self):
        result_dirs = [
            # "websites/all_models_with_split_websites_10s_pure_by_subject_weights_weight_none",
            # "websites/all_models_with_split_websites_10s_pure_by_subject_weights_weights_1_1",
            # "websites/all_models_with_split_websites_10s_pure_by_subject_weights_weights_1_5",
            # "websites/all_models_with_split_websites_10s_pure_by_video_weights_weight_none",
            "websites/all_models_with_split_websites_10s_pure_by_video_weights_weights_1_1",
            # "websites/all_models_with_split_websites_10s_pure_by_video_weights_weights_1_5",
        ]
        for result_dir in result_dirs:
            correct_metrics.Correcter(results_dir=result_dir).correct()

    def test_analyze_labels_in_subjects(self):
        analyze_labels_in_subjects_dist.Analyzer("samplesv2_2021-06-15_20:26:49").analyze()

    def test_analyze_metrics(self):
        result_dirs = [
            # best websites
            "websites/big/test_vgg_m_classic_less_layers_adam_lr_1e3_wd_1e1",

            # best charts
            "charts/big_duplicate/test_vgg_m_classic_drop1_adam_lr_1e5_wd_1e2"
        ]
        for result_dir in result_dirs:
            analyze_metrics.Analyzer(results_dir=f"{result_dir}").analyze()

    def test_analyze_frames(self):
        # results = [
        #     # best websites
        #     "websites/big/test_vgg_m_classic_less_layers_adam_lr_1e3_wd_1e1",
        # ]
        # for result in results:
        #     analyze_predicted_frames.Analyzer(
        #         result,
        #         {
        #             "00": 50,
        #             "01": 50,
        #             "10": 50,
        #             "11": 50
        #         }).analyze(iteration=5, combination=Combination("m_classic_less_layers", "vgg"))

        results = [
            # best charts
            "charts/big_duplicate/test_vgg_m_classic_drop1_adam_lr_1e5_wd_1e2"
        ]
        for result in results:
            analyze_predicted_frames.Analyzer(
                result,
                {
                    "00": 50,
                    "01": 50,
                    "10": 50,
                    "11": 50
                }).analyze(iteration=8, combination=Combination("m_classic_drop1", "vgg"))

    def test_analyze_samples(self):
        analyze_samples.Analyzer("websites_10s").analyze()

    def test_verify_frame(self):
        verify_frame.Verifier("17S2-PH1107-CY1307-LEC_20180220__Time1862.8915sec")

    def test_analyze_state_point_events(self):
        # StatePointAnalyzer().calculate_crossings()
        StatePointAnalyzer().save_crossing_frames()

    def test_take_screenshot(self):
        event: Event
        frames.save_frames([579, 668], C.StatePoint.GET_EVENT_FRAME_NAME)

    def test_compare_events(self):
        compare_frames.compare([574, 574])

    def test_visualize_video(self):
        visualize_state_events.visualize("17S1-PH1011-LEC_20171010")
