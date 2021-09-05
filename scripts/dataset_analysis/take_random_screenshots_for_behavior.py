from pathlib import Path

import pandas as pd

from scripts.definitions.domain.event import Event
from scripts.utils import frames
from scripts.utils.constants import C

OUTPUT_DIR = "outputs/beh_screens"


def check(beh_id):
    output_dir = f"{OUTPUT_DIR}/{beh_id}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    events_df = pd.read_csv(C.PROCESSED_ANNOTATIONS_PATH)
    events_df = events_df[events_df[C.A_BEH_ID] == beh_id]

    chosen_events_df = events_df.sample(n=50)
    chosen_events_ids = chosen_events_df[C.A_EVENT_ID].tolist()

    event: Event
    frames.save_frames(event_ids=chosen_events_ids,
                       output_file_provider=lambda event: f"{event.event_id}.jpg",
                       output_dir=output_dir,
                       points=[0.3])


if __name__ == "__main__":
    check(C.B_CHARTS_P)
