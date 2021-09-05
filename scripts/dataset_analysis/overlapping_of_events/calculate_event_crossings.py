"""
In this script we calculate:
- for every video
- for all behaviors that we're interested in
-- pick first behavior
-- get all it's state events
-- for every state event
--- pick another behavior
--- for all it's state events
--- calculate if "parent/first" state event overlaps with the current state event and if so - for how long

This calculation allows us to prepare charts describing the total number of collisions, e.g. between charts and images.
"""
from typing import List

import intervals as inter
import pandas as pd
from tqdm import tqdm

from scripts.dataset_analysis.overlapping_of_events.event_crossings import EventWithCrossings, CrossEvent
from scripts.definitions.domain.event import Event
from scripts.utils.constants import C


def create_event_crossings():
    # load all events of our interest (i.e. visual ones)
    events_df = pd.read_csv(C.VISUAL_EVENTS_PATH, converters=C.A_CONVERTERS)
    videos = events_df[C.A_VIDEO].unique()

    result = []

    for video in tqdm(videos):

        # get only events in this video
        video_df = events_df[events_df[C.A_VIDEO] == video]
        events: List[Event] = Event.from_df(video_df)

        for event_1 in events:

            event_1_start = event_1.start
            event_1_stop = event_1.stop

            event_with_crossings: EventWithCrossings = EventWithCrossings(
                event_id=event_1.event_id,
                beh_id=event_1.beh_id,
                interval=(event_1_start, event_1_stop),
                video=video)

            # let's analyze all other video events
            for event_2 in events:

                # skip current event
                if event_1.event_id == event_2.event_id:
                    continue

                event_2_start = event_2.start
                event_2_stop = event_2.stop

                is_cross = _is_cross(event_1_start, event_1_stop,
                                     event_2_start, event_2_stop)
                if not is_cross:
                    cross_event = CrossEvent(event_id=event_2.event_id,
                                             beh_id=event_2.beh_id,
                                             cross_interval=(event_2_start, event_2_stop),
                                             percent_of_parent_interval=None)
                    event_with_crossings.add_cross_event(cross_event=cross_event, is_cross=False)

                else:
                    percentage = _calculate_percentage_for_crossing_events(event_1_start, event_1_stop,
                                                                           event_2_start, event_2_stop)
                    cross_event = CrossEvent(event_id=event_2.event_id,
                                             beh_id=event_2.beh_id,
                                             cross_interval=(event_2_start, event_2_stop),
                                             percent_of_parent_interval=percentage)
                    event_with_crossings.add_cross_event(cross_event=cross_event, is_cross=True)

            result.append(event_with_crossings)

    return result


def _is_cross(start1, stop1, start2, stop2):
    is_not = start1 > stop2 or start2 > stop1
    return not is_not


def _calculate_percentage_for_crossing_events(start1, stop1, start2, stop2):
    duration1 = stop1 - start1
    # 1st event is point event
    if duration1 == 0:
        return 1

    duration2 = stop2 - start1
    # second event is point event
    if duration2 == 0:
        # let's return 0 here
        return 0

    interval1 = inter.closed(start1, stop1)
    interval2 = inter.closed(start2, stop2)

    common = interval1.intersection(interval2)
    common_duration = common.upper - common.lower

    return common_duration / duration1


if __name__ == "__main__":
    # calculate
    crossings = create_event_crossings()

    # save results
    EventWithCrossings.save(crossings, "new_crossings.pickle")

    # check if we can load results
    crossings = EventWithCrossings.load("new_crossings.pickle")
