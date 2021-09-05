"""
Plot labels distribution in sets in order to show the difference between 2 approaches:
- make sure all events from the same video are in the same set
- make sure all events from the same subejct are in the same set
"""
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scripts.prepare_data_for_training.sets_split.charts_images.ci_try_split_subjects_into_sets import ci_try_split_subjects_into_sets
from scripts.prepare_data_for_training.sets_split.charts_images.ci_try_split_videos_into_sets import ci_try_split_videos_into_sets
from scripts.prepare_data_for_training.sets_split.websites.web_try_split_subjects_into_sets import web_try_split_subjects_into_sets
from scripts.prepare_data_for_training.sets_split.websites.web_try_split_videos_into_sets import web_try_split_videos_into_sets
from scripts.utils.common import mkdir_path
from scripts.utils.constants import C


def add_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=30)


def plot_labels_in_sets(dist_in_sets, output):
    mkdir_path(output)

    sns.reset_orig()  # get default matplotlib styles back
    COLORS = sns.color_palette('tab10', n_colors=3)

    set_names = ["Train set", "Dev set", "Test set"]
    set_idx = np.arange(len(dist_in_sets))

    """
    Plot videos per set
    """
    fig, ax = plt.subplots(figsize=(16, 12))
    plt.tight_layout(pad=6)

    bar_width = 0.3
    bar_centers = set_idx

    ax.set_xticks(bar_centers)
    ax.set_xticklabels(set_names)
    ax.set_ylim([0, 2000])
    ax.tick_params(axis='y', which='major', labelsize=25)
    ax.tick_params(axis='x', which='major', labelsize=40)

    for set_index, set_dist in enumerate(dist_in_sets.values()):
        no_of_0 = set_dist["sum0"]
        no_of_1 = set_dist["sum1"]

        ax.bar(x=bar_centers[set_index] - bar_width / 2,
               width=bar_width,
               height=no_of_0,
               color="red",
               label="Label False")
        ax.bar(x=bar_centers[set_index] + bar_width / 2,
               width=bar_width,
               height=no_of_1,
               color="blue",
               label="Label True")

    add_legend(ax)
    # plt.show()
    plt.savefig(output)


if __name__ == "__main__":
    dist_in_sets_split_by_videos = web_try_split_videos_into_sets()
    dist_in_sets_split_by_subjects = web_try_split_subjects_into_sets()

    plot_labels_in_sets(web_try_split_videos_into_sets(),
                        output=f"{C.DistributionsAndSets.DISTRIBUTIONS_RESULTS}/websites/labels_distribution_in_sets_split_by_videos.png")
    plot_labels_in_sets(web_try_split_subjects_into_sets(),
                        output=f"{C.DistributionsAndSets.DISTRIBUTIONS_RESULTS}/websites/labels_distribution_in_sets_split_by_subjects.png")

    plot_labels_in_sets(ci_try_split_videos_into_sets(),
                        output=f"{C.DistributionsAndSets.DISTRIBUTIONS_RESULTS}/charts/labels_distribution_in_sets_split_by_videos.png")
    plot_labels_in_sets(ci_try_split_subjects_into_sets(),
                        output=f"{C.DistributionsAndSets.DISTRIBUTIONS_RESULTS}/charts/labels_distribution_in_sets_split_by_subjects.png")
