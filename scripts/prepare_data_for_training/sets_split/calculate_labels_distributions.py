from typing import List

from tqdm import tqdm

from scripts.prepare_data_for_training.samples.samples_v2 import SampleV2
from scripts.utils.common import pickle_obj
from scripts.utils.constants import C
from scripts.utils.frames import concat_frames


def check_dist_in_samples(samples_name):
    samples1 = SampleV2.from_csv(samples_name)
    print(check_dist_samples(samples1))


def check_dist_samples(samples: List[SampleV2]):
    dist = {
        0: 0,
        1: 0
    }
    for sample in samples:
        dist[sample.label] = dist[sample.label] + 1
    return dist


def prepare_data(
        samples,
        frames,
        global_output,
        video_output,
        subject_output):
    """
    Calculate and save what is the distribution of labels 0 and 1 in different contexts.

    :param samples: name of file with Samples
    :param frames: name of file with Frames
    :param global_output: output for global distribution
    :param video_output: output for distribution in each video
    :param subject_output: output for distribution in each subject
    """
    samples: List[SampleV2] = SampleV2.from_csv(samples)

    frames_df = concat_frames(frames)

    dist_global = {
        0: 0,
        1: 0
    }
    dist_per_video = {}
    dist_per_subject = {}

    for sample in tqdm(samples):
        frame_row = frames_df[frames_df[C.F_ID] == sample.id].iloc[0]

        video = frame_row[C.F_VIDEO]
        subject = C.SUBJECTS_BY_VIDEO[video]
        label = sample.label

        if video not in dist_per_video:
            dist_per_video[video] = {
                0: 0,
                1: 1
            }
        dist_per_video[video][label] = dist_per_video[video][label] + 1

        if subject not in dist_per_subject:
            dist_per_subject[subject] = {
                0: 0,
                1: 1
            }
        dist_per_subject[subject][label] = dist_per_subject[subject][label] + 1

        dist_global[label] = dist_global[label] + 1

    pickle_obj(dist_global, global_output)
    pickle_obj(dist_per_video, video_output)
    pickle_obj(dist_per_subject, subject_output)


if __name__ == "__main__":
    check_dist_in_samples("chart_images_pure")
    check_dist_in_samples("chart_images_5s_pure")

    prepare_data(samples="chart_images_pure",
                 frames=[C.Frames.POINT, C.Frames.CHART_IMAGE_S10_LABEL_0],
                 global_output=f"{C.DistributionsAndSets.DISTRIBUTIONS_RESULTS}/dist_chart_images/global.pickle",
                 video_output=f"{C.DistributionsAndSets.DISTRIBUTIONS_RESULTS}/dist_chart_images/video.pickle",
                 subject_output=f"{C.DistributionsAndSets.DISTRIBUTIONS_RESULTS}/dist_chart_images/subjects.pickle")
    prepare_data(samples="chart_images_5s_pure",
                 frames=[C.Frames.POINT, C.Frames.CHART_IMAGE_S10_LABEL_0],
                 global_output=f"{C.DistributionsAndSets.DISTRIBUTIONS_RESULTS}/dist_chart_images_10/global.pickle",
                 video_output=f"{C.DistributionsAndSets.DISTRIBUTIONS_RESULTS}/dist_chart_images_10/video.pickle",
                 subject_output=f"{C.DistributionsAndSets.DISTRIBUTIONS_RESULTS}/dist_chart_images_10/subjects.pickle")
