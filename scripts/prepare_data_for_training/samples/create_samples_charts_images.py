from typing import Optional

from tqdm import tqdm

from scripts.prepare_data_for_training.samples.samples_utils import get_behaviors_to_labels, \
    aggregate_dict_list_values, \
    get_frame_ids_that_have_relevant_behs
from scripts.prepare_data_for_training.samples.samples_v2 import *
from scripts.prepare_data_for_training.sets_split.calculate_labels_distributions import check_dist_in_samples
from scripts.utils.common import subtract_lists
from scripts.utils.constants import C
from scripts.utils.frames import concat_frames


def create_charts_samples(output_name: Optional[str],
                          labels_to_behaviors,
                          frames_df_paths) -> List[SampleV2]:
    frames_df = concat_frames(frames_df_paths)
    return create_website_samples_using_frames_df(output_name, labels_to_behaviors, frames_df)


def create_website_samples_using_frames_df(output_name: Optional[str],
                                           labels_to_behaviors,
                                           frames_df):
    """
    Samples specifically for charts & images
    """
    # invert dictionary so we have {'beh': 'label'}
    behaviors_to_labels = get_behaviors_to_labels(labels_to_behaviors)
    # get all behaviors that we're interested in
    labeled_behs = aggregate_dict_list_values(labels_to_behaviors)

    # this will be the result
    samples: List[SampleV2] = []

    # filter out all frames that dont have any relevant behaviors
    ids_of_relevant_frames = get_frame_ids_that_have_relevant_behs(frames_df, labeled_behs)
    frames_df = frames_df[frames_df[C.F_ID].isin(ids_of_relevant_frames)]

    videos = frames_df[C.F_VIDEO].unique()

    # for every video
    for video in tqdm(videos):

        video_frames = frames_df[frames_df[C.F_VIDEO] == video]

        for _, frame in video_frames.iterrows():

            beh_id = frame[C.F_BEH_ID]
            if beh_id not in labeled_behs:
                continue

            # we always trust point events
            if beh_id == C.B_CHARTS_P or beh_id == C.B_IMAGES_P:

                samples.append(SampleV2(
                    frame_id=frame[C.F_ID],
                    splits_names=frame[C.F_SPLITS],
                    label=behaviors_to_labels[beh_id]
                ))

            # ignore websites - they are recognized by other classifier
            elif beh_id == C.B_WEBSITE_P:
                continue

            else:
                # this doesn't need to be separated from handling websites but for now let's leave it
                samples.append(SampleV2(
                    frame_id=frame[C.F_ID],
                    splits_names=frame[C.F_SPLITS],
                    label=behaviors_to_labels[beh_id]
                ))

    if output_name:
        SampleV2.to_csv(samples, output_name)
    return samples


def generate_for_point_events():
    chart_images_samples_1 = create_charts_samples(output_name="chart_images_pure",
                                                   labels_to_behaviors={
                                                       0: subtract_lists(C.VISUAL_BEHS,
                                                                         [C.B_CHARTS_P, C.B_IMAGES_P]),
                                                       1: [C.B_CHARTS_P, C.B_IMAGES_P],
                                                   },
                                                   frames_df_paths=[
                                                       C.Frames.POINT]
                                                   )
    chart_images_samples_2 = SampleV2.from_csv("chart_images_pure")
    check_dist_in_samples("chart_images_pure")


def generate_for_point_events_and_generated_frames():
    chart_images_samples_1 = create_charts_samples(output_name="chart_images_5s_pure",
                                                   labels_to_behaviors={
                                                       0: subtract_lists(C.VISUAL_BEHS,
                                                                         [C.B_CHARTS_P, C.B_IMAGES_P]),
                                                       1: [C.B_CHARTS_P, C.B_IMAGES_P],
                                                   },
                                                   frames_df_paths=[
                                                       C.Frames.POINT,
                                                       C.Frames.CHART_IMAGE_S10_LABEL_0]
                                                   )
    chart_images_samples_2 = SampleV2.from_csv("chart_images_5s_pure")
    check_dist_in_samples("chart_images_5s_pure")


if __name__ == "__main__":
    generate_for_point_events()
    generate_for_point_events_and_generated_frames()
