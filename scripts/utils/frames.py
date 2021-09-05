from pathlib import Path

import pandas as pd
import torchvision as tv
from torchvision.transforms import transforms
from tqdm import tqdm

from scripts.definitions.domain.event import Event
from scripts.utils.constants import C

DEFAULT_POINTS = [0]


def concat_frames(frame_df_paths):
    frames_df = None
    for path in frame_df_paths:
        df = pd.read_csv(path, converters=C.F_CONVERTERS)
        if frames_df is None:
            frames_df = df
            continue
        else:
            frames_df = frames_df.append(df)
    return frames_df


def save_frames(event_ids,
                output_file_provider,
                output_dir,
                points=DEFAULT_POINTS):
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    events_df = pd.read_csv(C.PROCESSED_ANNOTATIONS_PATH, converters=C.A_CONVERTERS)

    for event_id in tqdm(event_ids):
        for point in points:
            event_row = events_df.loc[events_df[C.A_EVENT_ID] == event_id].iloc[0]
            event = Event.from_df_row(event_row)

            output_file = output_file_provider(event)
            output_path = f"{output_dir}/P_{point}_{output_file}"

            video_path = f"{C.ORIGINAL_VIDEOS_DIR}/{event.video}.mp4"
            duration = event.stop - event.start
            video = tv.io.video.read_video(video_path,
                                           start_pts=event.start + point * duration,
                                           end_pts=event.start + point * duration,
                                           pts_unit='sec')[0]
            frame = video[0]
            image = rgb_frame_to_image(frame)
            image.save(output_path)


def rgb_frame_to_image(frame_tensor):
    # convert H,W,C to C,H,W format
    frame_tensor = frame_tensor.permute([2, 0, 1])
    return transforms.ToPILImage()(frame_tensor)



if __name__ == "__main__":
    save_frames(event_ids=[682, 631, 632, 683, 635, 684, 636, 637],
                output_file_provider=C.StatePoint.GET_VID_EVENT_FRAME_NAME,
                output_dir=f"{C.StatePoint.RANDOM_FRAMES_DIR}/writing",
                points=[0, 0.1, 0.2, 0.5])
