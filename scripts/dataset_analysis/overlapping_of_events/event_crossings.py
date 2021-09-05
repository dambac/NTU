from typing import Tuple, List

from scripts.utils.common import pickle_obj, unpickle_obj
from scripts.utils.constants import C


class CrossEvent:
    """
    When analyzing which events collide with each other, we analyze events one by one. A parent event is an event
    that we're currently checking - we want to get all other video events and find out if they're crossing with parent
    event of not - those child events are represented by this class.
    """

    def __init__(self,
                 event_id,
                 beh_id,
                 cross_interval: Tuple[float, float],
                 percent_of_parent_interval):
        self.event_id = event_id
        self.beh_id = beh_id
        self.cross_interval = cross_interval
        self.cross_percentage = percent_of_parent_interval


class EventWithCrossings:
    """
    In thesis, we wanted to analyzed how many events collide with each other, e.g. how many chart events overlap with
    images events.
    This class, for every event in every video, represents its annotation's time interval and a list of all other
    video events that do / do not collide with it.
    """

    def __init__(self,
                 event_id,
                 beh_id,
                 interval: Tuple[float, float],
                 video):
        self.event_id: int = event_id
        self.beh_id: str = beh_id
        self.interval = interval
        self.not_crossing_events = []
        self.crossing_events: List[CrossEvent] = []
        self.video = video

    def add_cross_event(self, cross_event: CrossEvent, is_cross):
        if is_cross:
            self.crossing_events.append(cross_event)
        else:
            self.not_crossing_events.append(cross_event)

    @staticmethod
    def save(events, file_name):
        file_path = f"{C.EventCrossings.DF_OUTPUT_DIR}/{file_name}"
        pickle_obj(events, file_path)

    @staticmethod
    def load(file_name):
        file_path = f"{C.EventCrossings.DF_OUTPUT_DIR}/{file_name}"
        return unpickle_obj(file_path)

    def __repr__(self):
        return f"{self.event_id},{self.beh_id}"
