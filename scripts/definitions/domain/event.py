import datetime

from scripts.utils.constants import C


class Event:
    """
    Class representing a single Event.
    """
    def __init__(self,
                 event_id,
                 beh,
                 beh_id,
                 duration,
                 start,
                 stop,
                 video,
                 ra_id):
        self.event_id = event_id
        self.beh = beh
        self.beh_id = beh_id
        self.duration = duration
        self.start = start
        self.stop = stop
        self.video = video
        self.ra_id = ra_id

    @staticmethod
    def from_df_row(row):
        return Event(
            event_id=row[C.A_EVENT_ID],
            beh=row[C.A_BEHAVIOR],
            beh_id=row[C.A_BEH_ID],
            duration=row[C.A_DURATION],
            start=row[C.A_START],
            stop=row[C.A_STOP],
            video=row[C.A_VIDEO],
            ra_id=row[C.A_RA_ID]
        )

    @staticmethod
    def from_df(df):
        events = []
        for _, row in df.iterrows():
            events.append(Event.from_df_row(row))
        return events

    def __repr__(self):
        start_time = self.format_seconds(self.start)
        stop_time = self.format_seconds(self.stop)
        return f"{self.event_id}[{start_time}-{stop_time}] {self.beh}"

    @staticmethod
    def format_seconds(seconds):
        date = datetime.datetime.fromtimestamp(seconds) - datetime.timedelta(hours=1)
        return date.strftime("%H:%M:%S:%f")[:-3]
