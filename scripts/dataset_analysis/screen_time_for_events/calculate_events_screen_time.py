import math
import os
from typing import Dict

import intervals as inter
import pandas as pd
import torch
from tqdm import tqdm

from scripts.definitions.domain.event import Event
from scripts.dataset_analysis.screen_time_for_events.calculate_layout_for_each_video_chunk import read_layout_for_each_video_chunk, VideoChunkLayouts
from scripts.utils.common import pickle_obj, unpickle_obj, group_by
from scripts.utils.constants import C


def calculate_events_screen_time(output_file):
    """
    Check the amount of time events are simultaneously annotated with "Screen" layout type.
    """
    layouts_for_each_video_chunk = read_layout_for_each_video_chunk("layouts_for_each_video_chunk.pickle")
    chunk_layouts_for_videos: Dict[str, VideoChunkLayouts] = group_by(seq=layouts_for_each_video_chunk,
                                                                      key=lambda vcl: vcl.video)

    behs_of_interest = [C.B_WEBSITE_S, C.B_CHARTS_S, C.B_IMAGES_S]

    events_df = pd.read_csv(C.VISUAL_EVENTS_PATH, converters=C.A_CONVERTERS)
    events_df = events_df[events_df[C.A_BEH_ID].isin(behs_of_interest)]

    screen_time_for_behs = {}
    for beh in behs_of_interest:
        screen_time_for_behs[beh] = []

    for beh_id in behs_of_interest:

        beh_df = events_df[events_df[C.A_BEH_ID] == beh_id]
        events = Event.from_df(beh_df)

        for event in tqdm(events):
            video = event.video
            if video not in chunk_layouts_for_videos:
                continue
            video_chunk_layouts = chunk_layouts_for_videos[video][0]

            screen_time = get_screen_time_percentage(event.start, event.stop, video_chunk_layouts)
            screen_time_for_behs[beh_id].append(screen_time)

    pickle_obj(screen_time_for_behs, f"{C.EventCrossings.EVENT_LAYOUT_DISTRIBUTION_DIR}/{output_file}")
    z = 1


def get_screen_time_percentage(start, stop, video_chunk_layouts: VideoChunkLayouts):
    """
    Converted videos = picked every 10th frame, so 1 frame describes 10 original frames.
    In other words, if video has 100 chunks and chunk lasts 10 seconds, we assume that:
    - first chunk describes [0, 10)
    - second chunk describes [10, 20)
    - third chunk describes [20, 30)
    So for event [13, 27] we get:
    - 7 seconds for chunk 2
    - 7 seconds for chunk 3
    """
    screen_time = 0
    camera_time = 0

    event_interval = inter.closed(start, stop)
    chunk_duration = video_chunk_layouts.chunk_duration

    first_chunk = math.floor(start / chunk_duration)
    last_chunk = math.floor(stop / chunk_duration)
    available_number_of_chunks = len(video_chunk_layouts.layouts_for_chunks)

    for chunk in range(first_chunk, last_chunk):
        if chunk >= available_number_of_chunks:
            break
        chunk_start = chunk * chunk_duration
        chunk_stop = (chunk + 1) * chunk_duration
        chunk_interval = inter.closedopen(chunk_start, chunk_stop)

        event_chunk_intersection = chunk_interval.intersection(event_interval)
        event_chunk_length = event_chunk_intersection.upper - event_chunk_intersection.lower

        chunk_layout = video_chunk_layouts.layouts_for_chunks[chunk]
        is_screen = chunk_layout in C.SCREEN_LAYOUTS

        if is_screen:
            screen_time = screen_time + event_chunk_length
        else:
            camera_time = camera_time + event_chunk_length

    total_time = screen_time + camera_time
    screen_time_percentage = screen_time / total_time
    return screen_time_percentage


def get_video_layouts_list(video):
    layouts_path = f"{C.LAYOUT_ANNOTATIONS_DIR}/{video}.pt"
    if not os.path.exists(layouts_path):
        return None

    return torch.load(layouts_path).tolist()


def read_events_screen_time(input_file):
    return unpickle_obj(f"{C.EventCrossings.EVENT_LAYOUT_DISTRIBUTION_DIR}/{input_file}")


if __name__ == "__main__":
    calculate_events_screen_time("events_screen_time.pickle")
    layouts_for_each_video_chunk = read_events_screen_time("events_screen_time.pickle")
