"""
The results of this file is used in Jupyter notebook to generate more frames for charts
"""
from typing import List

import pandas as pd
from tqdm import tqdm

from scripts.dataset_analysis.overlapping_of_events.event_crossings import EventWithCrossings, CrossEvent
from scripts.definitions.domain.event import Event
from scripts.utils.common import group_by, pickle_obj
from scripts.utils.constants import C


def event_does_not_cross_with_any_of_behaviors(crossing: EventWithCrossings, forbidden_behaviors):
    cross_event: CrossEvent
    for cross_event in crossing.crossing_events:
        if cross_event.beh_id in forbidden_behaviors:
            return False
    return True


def remove_duplicates_from_crossings(crossings: List[EventWithCrossings],
                                     events: List[Event]):
    results = []
    events_by_event_id = group_by(seq=events, key=lambda event: event.event_id)
    crossings_by_video = group_by(seq=crossings, key=lambda cross: cross.video)

    for video, video_crossings in tqdm(crossings_by_video.items()):
        occupied_timestamps = []

        for cross in video_crossings:

            cross_event: Event = events_by_event_id[cross.event_id][0]
            timestamp = (cross_event.start + cross_event.stop) / 2

            has_collision = False
            for occupied_time in occupied_timestamps:
                if abs(timestamp - occupied_time) < 10:
                    has_collision = True
                    break

            if not has_collision:
                results.append(cross)

                occupied_timestamps.append(timestamp)

    return results


if __name__ == "__main__":
    print("loading events")
    visual_events_df = pd.read_csv(C.VISUAL_EVENTS_PATH, converters=C.A_CONVERTERS)
    visual_events: List[Event] = Event.from_df(visual_events_df)

    print("loading crossings")
    crossings: List[EventWithCrossings] = EventWithCrossings.load("new_crossings.pickle")

    # since we want to add label 0, we care only about other visual behaviors - we should exclude
    # charts, images & websites because we already have label 1 for them
    already_handled_behs = [C.B_CHARTS_P, C.B_CHARTS_S, C.B_IMAGES_P, C.B_IMAGES_S, C.B_WEBSITE_P, C.B_WEBSITE_S]
    crossings_other = [cross for cross in crossings if cross.beh_id not in already_handled_behs]
    crossings_other_pure = [cross for cross in crossings_other if
                            event_does_not_cross_with_any_of_behaviors(cross, already_handled_behs)]

    crossings_other_no_duplicates = remove_duplicates_from_crossings(crossings_other_pure,
                                                                     visual_events)

    event_ids = [cross.event_id for cross in crossings_other_no_duplicates]
    pickle_obj(event_ids, "/tmp/event_ids_for_new_chart_image_label_0_frames.pickle")
