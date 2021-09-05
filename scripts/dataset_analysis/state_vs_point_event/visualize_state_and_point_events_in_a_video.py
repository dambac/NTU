# Import data
import math
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from moviepy.video.io.VideoFileClip import VideoFileClip

from scripts.definitions.domain.event import Event
from scripts.utils.constants import C

TICK_SEGMENT_IN_SECONDS = 15 * 60
PRECISION_IN_SECONDS = 1


def get_beh_ids_sorted(events: List[Event]):
    ids = list(set([e.beh_id for e in events]))
    return sorted(ids)


def add_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=20)


def visualize(video, output_file):
    # get video duration
    video_file = f"{C.ORIGINAL_VIDEOS_DIR}/{video}.mp4"
    video_clip = VideoFileClip(video_file)
    video_duration_seconds = video_clip.duration

    events_df = pd.read_csv(C.VISUAL_EVENTS_PATH)

    video_df = events_df[events_df[C.A_VIDEO] == video]

    point_events = list(C.POINT_STATE_BEH_MAP.keys())
    state_events = list(C.POINT_STATE_BEH_MAP.values())
    video_df = video_df[video_df[C.A_BEH_ID].isin(point_events + state_events)]

    events: List[Event] = Event.from_df(video_df)

    ra_ids = list(set([state_event.ra_id for state_event in events]))
    ra_ids = sorted(ra_ids)

    sns.reset_orig()  # get default matplotlib styles back
    COLORS = sns.color_palette('tab10', n_colors=len(ra_ids))

    colors_for_ra = {}
    for index, ra_id in enumerate(ra_ids):
        colors_for_ra[ra_id] = COLORS[index]

    # Figure and Axes
    fig, ax = plt.subplots(1, 1, figsize=(20, 8), facecolor='#f7f7f7', dpi=200)

    # X range & ticks
    number_of_segments = math.ceil(video_duration_seconds / TICK_SEGMENT_IN_SECONDS)
    """
    Artificially decrease for particular video
    """
    # number_of_segments = 2
    plot_duration = number_of_segments * TICK_SEGMENT_IN_SECONDS

    x_second_indexes = list(range(0, plot_duration, TICK_SEGMENT_IN_SECONDS))

    x_ticks = x_second_indexes
    x_ticks_labels = []
    for seconds_timestamp in x_second_indexes:
        t = datetime(2020, 1, 1)
        t = t + timedelta(seconds=seconds_timestamp)
        label = t.strftime("%H:%M:%S")
        x_ticks_labels.append(label)

    for seg_index in range(number_of_segments):
        lines_position = seg_index * TICK_SEGMENT_IN_SECONDS
        ax.vlines(x=lines_position, ymin=0, ymax=26, color='black', alpha=1, linewidth=1, linestyles='dotted')

    # Y range & ticks
    beh_ids = get_beh_ids_sorted(events)
    beh_titles = {beh_id: C.BEH_SHORT_LABELS[beh_id] for beh_id in beh_ids}
    number_of_behs = len(beh_ids)

    # we leave 0 and last index
    y_ticks = list(range(1, number_of_behs + 1))
    y_ticks_labels = list(beh_titles.values())

    beh_id_and_ra_id_to_y_position = dict()
    for beh_index, beh_id in enumerate(beh_ids):
        for ra_index, ra_id in enumerate(ra_ids):
            beh_id_and_ra_id_to_y_position[(beh_id, ra_id)] = beh_index + 1 + ra_index * 0.1

    EVENT_ID_POSTIONS = {
        # 573: (0, 0),
        # 574: (0, 0),
        #
        # 608: (0, 0),
        # 609: (0, 0),

        # 613: (0, 0.1),
        # 577: (0, -0.3),
        #
        # 612: (0, 0.1),
        # 576: (0, -0.3),
        #
        # 664: (0, 0.1),
        # 615: (0, -0.3),
        #
        # 663: (0, 0.1),
        # 614: (0, -0.3),
        #
        # 616: (0, 0.1),
        # 617: (0, 0.1),

        # 1677: (0,0)
    }
    # Y lines
    for beh_id in beh_ids:
        for event in events:
            if event.beh_id != beh_id:
                continue
            y_position = beh_id_and_ra_id_to_y_position[(beh_id, event.ra_id)]

            start_second = math.floor(event.start) - 3
            stop_second = math.ceil(event.stop) + 3
            color = colors_for_ra[event.ra_id]
            plt.hlines(y_position, start_second, stop_second, lw=3, color=color, label=event.ra_id)

            if len(EVENT_ID_POSTIONS) > 0:
                if event.event_id not in EVENT_ID_POSTIONS:
                    continue
                move_left, move_top = EVENT_ID_POSTIONS[event.event_id]
            else:
                move_left, move_top = 0, 0
            middle = (start_second + stop_second) / 2
            # plt.text(x=middle - move_left, y=y_position + move_top, s=str(event.event_id), fontsize=15)

    ax.set_facecolor('#FFF0CB')
    ax.set_title(f"Annotations analysis for video: {video}", fontdict={'size': 18})
    ax.set(xlim=(0, plot_duration), ylim=(0, number_of_behs + 1), xlabel="Timestamp")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks_labels)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks_labels)

    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.xaxis.label.set_fontsize(20)

    add_legend(ax)

    plt.subplots_adjust(left=0.1, right=0.99)

    print(f"Writing to: {output_file}")
    plt.savefig(output_file)


if __name__ == "__main__":
    video = "15S1-PH1012-LEC_20150812"
    visualize(video, output_file=f"{C.StatePoint.CHARTS_OUTPUT_DIR}/{video}.png")
