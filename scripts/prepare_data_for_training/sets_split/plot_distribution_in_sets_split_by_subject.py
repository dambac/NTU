"""
The goal of those plots is to present how many labels 0 and 1 are in each set when we choose sets splitting approach:
"make sure all events coming from the same subject are in the same set".
"""
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scripts.prepare_data_for_training.sets_split.websites.web_try_split_subjects_into_sets import web_try_split_subjects_into_sets
from scripts.utils.common import unpickle_obj
from scripts.utils.constants import C


def add_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=30)


def plot_subjects_split(labels_distribution_in_videos_pickle,
                        set_split_results_dict,
                        output_file):
    """
    Plot distribution of labels in each subject.

    :param labels_distribution_in_videos_pickle: previously prepared distribution of labels in each video
    :param set_split_results_dict: dictionary telling which subject should be in which set
    :param output_file: where to save the results
    """
    videos_dist = unpickle_obj(labels_distribution_in_videos_pickle)

    train_subjects = set_split_results_dict["train"]["subjects"]
    dev_subjects = set_split_results_dict["dev"]["subjects"]
    test_subjects = set_split_results_dict["test"]["subjects"]

    sns.reset_orig()  # get default matplotlib styles back
    COLORS = sns.color_palette('deep', n_colors=3)

    set_names = ["Set 1", "Set 2", "Set 3"]
    sets_results = [
        train_subjects,
        dev_subjects,
        test_subjects
    ]

    subjects = list(C.SUBJECTS_DICT.keys())
    subject_indexes = {}
    for idx, subject in enumerate(subjects):
        subject_indexes[subject] = idx

    subjects_no = len(subjects)
    subjects_idx = np.arange(subjects_no)

    """
    Plot videos per set
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    plt.tight_layout(pad=3)

    bar_width = 0.3
    bar_centers = subjects_idx

    ax.set_xticks(bar_centers)
    ax.set_xticklabels(subjects_idx + 1)
    ax.tick_params(axis='y', which='major', labelsize=40)
    ax.tick_params(axis='x', which='major', labelsize=25)

    for set_index, set_subjects in enumerate(sets_results):
        color = COLORS[set_index]
        label = set_names[set_index]

        for subject_index, subject in enumerate(subjects):
            no_of_videos = 0
            if subject in set_subjects:
                no_of_videos = len([video for video in C.SUBJECTS_DICT[subject] if video in videos_dist])
            ax.bar(x=bar_centers[subject_index],
                   width=bar_width,
                   height=no_of_videos,
                   color=color,
                   label=label)
    add_legend(ax)
    # plt.show()
    plt.savefig(output_file)


def init_videos_per_subject(dist):
    videos_per_subject = {}
    for subject in C.SUBJECTS_DICT.keys():
        videos_per_subject[subject] = 0
    for video in dist["videos"]:
        subject = C.SUBJECTS_BY_VIDEO[video]
        videos_per_subject[subject] = videos_per_subject[subject] + 1
    return videos_per_subject


if __name__ == "__main__":
    plot_subjects_split(labels_distribution_in_videos_pickle=f"{C.DistributionsAndSets.DISTRIBUTIONS_RESULTS}/dist_websites_10/video.pickle",
                        set_split_results_dict=web_try_split_subjects_into_sets(),
                        output_file=f"{C.DistributionsAndSets.DISTRIBUTIONS_RESULTS}/websites/distribution_in_sets_split_by_subject.png")

    plot_subjects_split(labels_distribution_in_videos_pickle=f"{C.DistributionsAndSets.DISTRIBUTIONS_RESULTS}/dist_chart_images_10/video.pickle",
                        set_split_results_dict=web_try_split_subjects_into_sets(),
                        output_file=f"{C.DistributionsAndSets.DISTRIBUTIONS_RESULTS}/charts/distribution_in_sets_split_by_subject.png")