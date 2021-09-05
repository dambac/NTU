"""
Plot distribution of labels in every video in order to show the results of adding more samples from state events.
"""
from collections import OrderedDict
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from scripts.utils.common import unpickle_obj, mkdir_path
from scripts.utils.constants import C


def plot_video(global_pickle,
               video_pickle,
               output):
    global_dist = unpickle_obj(global_pickle)
    print(f"No of 0: {global_dist[0]}")
    print(f"No of 1: {global_dist[1]}")

    dist: Dict = unpickle_obj(video_pickle)
    videos = list(dist.keys())
    videos_no = len(videos)
    videos_idx = np.arange(videos_no)

    fig, ax = plt.subplots(figsize=(24, 12))
    plt.tight_layout(pad=5)

    bar_width = 0.3
    bar_gap = 3.5
    bar_centers = videos_idx + bar_gap / 2

    ax.set_xticks(bar_centers)
    ax.set_xticklabels([""] * videos_no)
    ax.tick_params(axis='both', which='major', labelsize=36)
    # ax.set_xticklabels([f"Video {idx}" for idx in videos_idx], rotation=60)

    for idx in videos_idx:
        video = videos[idx]
        video_dist = dist[video]

        ax.bar(x=bar_centers[idx] - bar_width / 2, width=bar_width, height=video_dist[0], label="Label False", color="red")
        ax.bar(x=bar_centers[idx] + bar_width / 2, width=bar_width, height=video_dist[1], label="Label True", color="blue")
        add_legend(ax)

    # plt.show()

    output_file = f"{C.DistributionsAndSets.DISTRIBUTIONS_RESULTS}/{output}"
    mkdir_path(output_file)
    plt.savefig(output_file)


def add_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=30)


if __name__ == "__main__":
    plot_video(global_pickle=f"{C.DistributionsAndSets.DISTRIBUTIONS_RESULTS}/dist_websites/global.pickle",
               video_pickle=f"{C.DistributionsAndSets.DISTRIBUTIONS_RESULTS}/dist_websites/video.pickle",
               output="labels_dist_website_video.png")
    plot_video(global_pickle=f"{C.DistributionsAndSets.DISTRIBUTIONS_RESULTS}/dist_websites_10/global.pickle",
               video_pickle=f"{C.DistributionsAndSets.DISTRIBUTIONS_RESULTS}/dist_websites_10/video.pickle",
               output="labels_dist_website_10_video.png")

    plot_video(global_pickle=f"{C.DistributionsAndSets.DISTRIBUTIONS_RESULTS}/dist_chart_images/global.pickle",
               video_pickle=f"{C.DistributionsAndSets.DISTRIBUTIONS_RESULTS}/dist_chart_images/video.pickle",
               output="labels_dist_chart_images_video.png")
    plot_video(global_pickle=f"{C.DistributionsAndSets.DISTRIBUTIONS_RESULTS}/dist_chart_images_10/global.pickle",
               video_pickle=f"{C.DistributionsAndSets.DISTRIBUTIONS_RESULTS}/dist_chart_images_10/video.pickle",
               output="labels_dist_chart_images_10_video.png")
