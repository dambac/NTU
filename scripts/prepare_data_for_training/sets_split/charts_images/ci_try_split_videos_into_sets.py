from scripts.utils.common import unpickle_obj
from scripts.utils.constants import C


def ci_try_split_videos_into_sets():
    """
    Split samples into sets using algorithm similar to the one for partition problem.
    """
    distributions_in_videos = unpickle_obj(f"{C.DistributionsAndSets.DISTRIBUTIONS_RESULTS}/dist_chart_images_10/video.pickle")
    videos_sorted_by_1_label = {k: v for k, v in
                                sorted(distributions_in_videos.items(), key=lambda item: item[1][1], reverse=True)}

    total_no_of_0 = sum([dist_in_video[0] for dist_in_video in distributions_in_videos.values()])
    total_no_of_1 = sum([dist_in_video[1] for dist_in_video in distributions_in_videos.values()])

    ratio = [.5, .3, .2]

    results = {
        "train": {
            "videos": [],
            "len": 0,
            "sum0": 0,
            "sum1": 0,
            "target0": ratio[0] * total_no_of_0,
            "target1": ratio[0] * total_no_of_1
        },
        "dev": {
            "videos": [],
            "len": 0,
            "sum0": 0,
            "sum1": 0,
            "target0": ratio[1] * total_no_of_0,
            "target1": ratio[1] * total_no_of_1
        },
        "test": {
            "videos": [],
            "len": 0,
            "sum0": 0,
            "sum1": 0,
            "target0": ratio[2] * total_no_of_0,
            "target1": ratio[2] * total_no_of_1
        }
    }

    for video, dist in videos_sorted_by_1_label.items():
        no_of_0 = dist[0]
        no_of_1 = dist[1]

        set_with_biggest_gap = max(results.items(), key=lambda item: item[1]["target1"] - item[1]["sum1"])[0]
        results[set_with_biggest_gap]["videos"].append(video)
        results[set_with_biggest_gap]["len"] = results[set_with_biggest_gap]["len"] + 1
        results[set_with_biggest_gap]["sum0"] = results[set_with_biggest_gap]["sum0"] + no_of_0
        results[set_with_biggest_gap]["sum1"] = results[set_with_biggest_gap]["sum1"] + no_of_1

    return results
