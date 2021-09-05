from scripts.prepare_data_for_training.sets_split.charts_images.ci_try_split_subjects_into_sets import ci_try_split_subjects_into_sets
from scripts.prepare_data_for_training.sets_split.charts_images.ci_try_split_videos_into_sets import ci_try_split_videos_into_sets
from scripts.prepare_data_for_training.samples.samples_v2 import SampleV2
from scripts.utils.constants import C
from scripts.utils.serialization import save_json, read_json


def invert_dict(original_dict):
    inverted_dict = {}
    for key, values in original_dict.items():
        for value in values:
            inverted_dict[value] = key
    return inverted_dict


def split_samples_to_sets_by_their_video(samples_name,
                                         set_to_videos,
                                         result_name):
    """
    :param samples_name: name of file with Samples
    :param set_to_videos: key = set name, value = videos that should belong to this set
    :param result_name: name of SamplesSplit
    :return:
    """
    samples = SampleV2.from_csv(samples_name)
    video_to_set = invert_dict(set_to_videos)

    train = []
    dev = []
    test = []

    sample: SampleV2
    for sample in samples:
        video = sample.extract_video_name()
        target_set = video_to_set[video]

        if target_set == "train":
            train.append(sample.id)
        if target_set == "dev":
            dev.append(sample.id)
        if target_set == "test":
            test.append(sample.id)

    samples_split = {
        "train": train,
        "dev": dev,
        "test": test
    }
    save_json(samples_split, f"{C.DistributionsAndSets.SETS_SPLITS_SAMPLES}/{result_name}.json")


def split_samples_to_sets_by_their_subject(samples_name,
                                           set_to_subjects,
                                           result_name):
    """
    :param samples_name: name of file with Samples
    :param set_to_subjects: key = set name, value = videos that should belong to this set
    :param result_name: name of SamplesSplit
    :return:
    """
    samples = SampleV2.from_csv(samples_name)
    subject_to_set = invert_dict(set_to_subjects)

    train = []
    dev = []
    test = []

    sample: SampleV2
    for sample in samples:
        video = sample.extract_video_name()
        subject = C.SUBJECTS_BY_VIDEO[video]
        target_set = subject_to_set[subject]

        if target_set == "train":
            train.append(sample.id)
        if target_set == "dev":
            dev.append(sample.id)
        if target_set == "test":
            test.append(sample.id)

    samples_split = {
        "train": train,
        "dev": dev,
        "test": test
    }
    save_json(samples_split, f"{C.DistributionsAndSets.SETS_SPLITS_SAMPLES}/{result_name}.json")


def save_which_video_and_subject_is_in_which_set():
    """
    Save which videos/subjects should belong to which set.
    """
    split_of_videos_into_sets = ci_try_split_videos_into_sets()
    split_of_videos = {
        "train": split_of_videos_into_sets["train"]["videos"],
        "dev": split_of_videos_into_sets["dev"]["videos"],
        "test": split_of_videos_into_sets["test"]["videos"]
    }
    save_json(split_of_videos, f"{C.DistributionsAndSets.SETS_SPLITS}/charts_videos_for_sets.json")

    split_of_subjects_into_sets = ci_try_split_subjects_into_sets()
    split_of_subjects = {
        "train": split_of_subjects_into_sets["train"]["subjects"],
        "dev": split_of_subjects_into_sets["dev"]["subjects"],
        "test": split_of_subjects_into_sets["test"]["subjects"]
    }
    save_json(split_of_subjects, f"{C.DistributionsAndSets.SETS_SPLITS}/charts_subjects_for_sets.json")


if __name__ == "__main__":
    # uncomment it once
    # save_which_video_and_subject_is_in_which_set()

    # read what we've calculated right above
    set_to_videos = read_json(f"{C.DistributionsAndSets.SETS_SPLITS}/charts_videos_for_sets.json")
    set_to_subjects = read_json(f"{C.DistributionsAndSets.SETS_SPLITS}/charts_subjects_for_sets.json")

    split_samples_to_sets_by_their_video(samples_name="chart_images_5s_pure",
                                         set_to_videos=set_to_videos,
                                         result_name="chart_images_5s_pure_by_video")
    split_samples_to_sets_by_their_subject(samples_name="chart_images_5s_pure",
                                           set_to_subjects=set_to_subjects,
                                           result_name="chart_images_5s_pure_by_subject")
