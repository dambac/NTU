"""
Authors of previous Bachelor's thesis annotated each video with a layout type manually. As it's not feasible
to annotate every single frame, annotations regard only a subset of all frames. We call time ranges between all
annotations a "chunk" and we assume that the annotation for n-th frame annotates whole [n, n+1] chunk. Here, we
calculate what is the annotation for every chunk.
"""
import os

import torch
from tqdm import tqdm

from scripts.utils.common import pickle_obj, unpickle_obj
from scripts.utils.constants import C


class VideoChunkLayouts:
    """
    All videos with their chunks and chunks' annotations.
    """

    def __init__(self,
                 video,
                 duration,
                 number_of_chunks,
                 chunk_duration,
                 layouts_for_chunks):
        self.video = video
        self.duration = duration
        self.number_of_chunks = number_of_chunks
        self.chunk_duration = chunk_duration
        self.layouts_for_chunks = layouts_for_chunks


def calculate_layout_for_each_video_chunk(output_file):
    # load previously calculated video durations
    video_durations = read_layout_for_each_video_chunk("video_durations.pickle")
    videos = list(video_durations.keys())

    video_layout_chunks = []
    for video in tqdm(videos):

        # get annotations for video frames
        video_layouts = get_video_layouts_list(video)
        if video_layouts is None:
            print(f"omitting video: {video}")
            continue

        number_of_chunks = len(video_layouts)
        video_duration = video_durations[video]
        chunk_duration = video_duration / number_of_chunks

        layouts_for_chunks = {}
        for chunk_index, chunk_layout in enumerate(video_layouts):
            layouts_for_chunks[chunk_index] = chunk_layout

        video_layout_chunks.append(VideoChunkLayouts(
            video,
            video_duration,
            number_of_chunks,
            chunk_duration,
            layouts_for_chunks
        ))

    pickle_obj(video_layout_chunks, f"{C.EventCrossings.EVENT_LAYOUT_DISTRIBUTION_DIR}/{output_file}")


def get_video_layouts_list(video):
    layouts_path = f"{C.LAYOUT_ANNOTATIONS_DIR}/{video}.pt"
    if not os.path.exists(layouts_path):
        return None

    return torch.load(layouts_path).tolist()


def read_layout_for_each_video_chunk(input_file):
    return unpickle_obj(f"{C.EventCrossings.EVENT_LAYOUT_DISTRIBUTION_DIR}/{input_file}")


if __name__ == "__main__":
    calculate_layout_for_each_video_chunk("layouts_for_each_video_chunk.pickle")
    layouts_for_each_video_chunk = read_layout_for_each_video_chunk("layouts_for_each_video_chunk.pickle")
