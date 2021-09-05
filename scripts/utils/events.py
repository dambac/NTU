import pandas as pd

from scripts.definitions.domain.event import Event
from scripts.utils.constants import C


def do_intervals_cross(start1, stop1, start2, stop2):
    does_not_cross = start1 > stop2 or stop1 < start2
    return not does_not_cross


def copy_df(df, data):
    new_df = pd.DataFrame(data=data, columns=df.columns)
    return new_df


def get_events_by_ids(events_df, event_ids):
    if type(event_ids) != list:
        event_ids = [event_ids]
    df = events_df[events_df[C.A_EVENT_ID].isin(event_ids)]
    return [Event.from_df_row(row) for _, row in df.iterrows()]
