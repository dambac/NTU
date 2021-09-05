import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm

from scripts.utils.common import pickle_obj, unpickle_obj
from scripts.utils.constants import C


def calculate_videos_durations(output_file):
    """
    Get duration of each video in seconds and save the result dict in file.
    """
    video_durations = {}

    events_df = pd.read_csv(C.PROCESSED_ANNOTATIONS_PATH, converters=C.A_CONVERTERS)
    video_names = events_df[C.A_VIDEO].unique()

    for video in tqdm(video_names):
        video_path = f"{C.ORIGINAL_VIDEOS_DIR}/{video}.mp4"
        video_clip = VideoFileClip(video_path)

        video_durations[video] = video_clip.duration

    pickle_obj(video_durations, f"{C.EventCrossings.EVENT_LAYOUT_DISTRIBUTION_DIR}/{output_file}")


def read_videos_durations(input_file):
    return unpickle_obj(f"{C.EventCrossings.EVENT_LAYOUT_DISTRIBUTION_DIR}/{input_file}")


if __name__ == "__main__":
    calculate_videos_durations("video_durations.pickle")
    video_durations = read_videos_durations("video_durations.pickle")
