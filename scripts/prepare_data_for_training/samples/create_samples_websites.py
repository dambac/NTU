from typing import Optional

from tqdm import tqdm

from scripts.dataset_analysis.overlapping_of_events.event_crossings import EventWithCrossings
from scripts.prepare_data_for_training.samples.samples_utils import get_behaviors_to_labels, \
    aggregate_dict_list_values, \
    get_frame_ids_that_have_relevant_behs
from scripts.prepare_data_for_training.samples.samples_v2 import *
from scripts.utils.common import subtract_lists, index_list_to_dict
from scripts.utils.constants import C
from scripts.utils.frames import concat_frames


def create_website_samples(output_name: Optional[str],
                           labels_to_behaviors,
                           frames_df_paths) -> List[SampleV2]:
    frames_df = concat_frames(frames_df_paths)
    return create_website_samples_using_frames_df(output_name, labels_to_behaviors, frames_df)


def create_website_samples_using_frames_df(output_name: Optional[str],
                                           labels_to_behaviors,
                                           frames_df) -> List[SampleV2]:
    """
    Samples specifically for websites
    """
    behaviors_to_labels = get_behaviors_to_labels(labels_to_behaviors)
    labeled_behs = aggregate_dict_list_values(labels_to_behaviors)

    # this will be the result
    samples: List[SampleV2] = []

    # filter out all frames that does not have any relevant behaviors
    ids_of_relevant_frames = get_frame_ids_that_have_relevant_behs(frames_df, labeled_behs)
    frames_df = frames_df[frames_df[C.F_ID].isin(ids_of_relevant_frames)]

    videos = frames_df[C.F_VIDEO].unique()

    event_crossings: List[EventWithCrossings] = EventWithCrossings.load("new_crossings.pickle")
    event_crossings_dict = index_list_to_dict(event_crossings,
                                              key_provider=lambda event: event.event_id)

    # for every video
    for video in tqdm(videos):

        video_frames = frames_df[frames_df[C.F_VIDEO] == video]

        for _, frame in video_frames.iterrows():

            beh_id = frame[C.F_BEH_ID]
            if beh_id not in labeled_behs:
                continue

            event_id = frame[C.F_EVENT_ID]

            if beh_id == C.B_WEBSITE_P or beh_id == C.B_WEBSITE_S:

                samples.append(SampleV2(
                    frame_id=frame[C.F_ID],
                    splits_names=frame[C.F_SPLITS],
                    label=behaviors_to_labels[beh_id]
                ))

            else:
                is_crossing_with_website = False
                for cross_event_id, cross_beh in event_crossings_dict[event_id].event_beh_ids:
                    if cross_beh == C.B_WEBSITE_P or cross_beh == C.B_WEBSITE_S:
                        is_crossing_with_website = True
                        break

                if not is_crossing_with_website:
                    samples.append(SampleV2(
                        frame_id=frame[C.F_ID],
                        splits_names=frame[C.F_SPLITS],
                        label=behaviors_to_labels[beh_id]
                    ))

    SampleV2.to_csv(samples, output_name)
    return samples


def prepare1():
    behs_1 = [C.B_WEBSITE_P, C.B_WEBSITE_S]
    website_samples_1 = create_website_samples(output_name="websites_10s_duplicates",
                                               labels_to_behaviors={
                                                   0: subtract_lists(C.SELECTED_POINT_BEHS,
                                                                     behs_1),
                                                   1: behs_1,
                                               },
                                               frames_df_paths=[
                                                   C.Frames.POINT_DUPLICATES,
                                                   C.Frames.WEBSITE_S10_DUPLICATES])
    website_samples_2 = SampleV2.from_csv("websites_10s_duplicates")


def prepare2():
    behs_1 = [C.B_WEBSITE_P, C.B_WEBSITE_S]
    website_samples_1 = create_website_samples(output_name="websites_10s",
                                               labels_to_behaviors={
                                                   0: subtract_lists(C.SELECTED_POINT_BEHS,
                                                                     behs_1),
                                                   1: behs_1,
                                               },
                                               frames_df_paths=[
                                                   C.Frames.POINT_DUPLICATES,
                                                   C.Frames.WEBSITE_S10])
    website_samples_2 = SampleV2.from_csv("websites_10s")


def prepare3():
    behs_1 = [C.B_WEBSITE_P, C.B_WEBSITE_S]
    website_samples_1 = create_website_samples(output_name="websites_10s_pure",
                                               labels_to_behaviors={
                                                   0: subtract_lists(C.SELECTED_POINT_BEHS,
                                                                     behs_1),
                                                   1: behs_1,
                                               },
                                               frames_df_paths=[
                                                   C.Frames.POINT,
                                                   C.Frames.WEBSITE_S10])
    website_samples_2 = SampleV2.from_csv("websites_10s_pure")


if __name__ == "__main__":
    # prepare1()
    # prepare2()
    prepare3()
