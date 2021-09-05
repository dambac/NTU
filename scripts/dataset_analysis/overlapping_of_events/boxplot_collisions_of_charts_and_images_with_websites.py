"""
Here we are making plots about the collisions between websites and charts + images.
"""
from typing import List

import matplotlib.pyplot as plt

from scripts.dataset_analysis.overlapping_of_events.event_crossings import EventWithCrossings, CrossEvent
from scripts.utils.constants import C


def plot_collisions_time_for_website(behs_with_labels, output_file):
    crossings: List[EventWithCrossings] = EventWithCrossings.load("new_crossings.pickle")
    crossings = [cross for cross in crossings if cross.beh_id == C.B_WEBSITE_S]

    fig, ax = plt.subplots()

    for index, (beh, label) in enumerate(behs_with_labels.items()):
        cross_percentages = get_website_cross_with_beh_percentages(crossings, beh)
        cross_percentages = [t * 100 for t in cross_percentages]
        ax.boxplot(cross_percentages, positions=[index])

    ax.set_xticklabels(behs_with_labels.values())
    plt.ylabel('Part of event that is overlapping (%)', fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=12)
    plt.savefig(output_file)
    # plt.show()


def get_website_cross_with_beh_percentages(crossings: List[EventWithCrossings], beh_id):
    result = []
    for cross in crossings:
        crossing_events: List[CrossEvent] = cross.crossing_events
        crossing_events = [ce for ce in crossing_events if ce.beh_id == beh_id]
        percentages = [ce.cross_percentage for ce in crossing_events]
        if len(percentages) > 0:
            result.extend(percentages)
    return result


def plot_collisions_time_with_website(behs_with_labels, output_file):
    crossings: List[EventWithCrossings] = EventWithCrossings.load("new_crossings.pickle")

    fig, ax = plt.subplots()

    for index, (beh, label) in enumerate(behs_with_labels.items()):
        beh_crossings = [cross for cross in crossings if cross.beh_id == beh]
        cross_percentages = get_cross_with_website_percentages(beh_crossings)
        cross_percentages = [t * 100 for t in cross_percentages]
        ax.boxplot(cross_percentages, positions=[index])

    ax.set_xticklabels(behs_with_labels.values())
    plt.ylabel('Part of event that is overlapping (%)', fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=12)

    plt.savefig(output_file)
    # plt.show()


def get_cross_with_website_percentages(crossings: List[EventWithCrossings]):
    result = []
    for cross in crossings:
        crossing_events: List[CrossEvent] = cross.crossing_events
        crossing_events = [ce for ce in crossing_events if ce.beh_id == C.B_WEBSITE_S]
        percentages = [ce.cross_percentage for ce in crossing_events]
        if len(percentages) > 0:
            result.extend(percentages)
    return result


if __name__ == "__main__":
    plot_collisions_time_with_website(
        behs_with_labels={
            C.B_CHARTS_S: "S, Charts in slides",
            C.B_IMAGES_S: "S, Images in slides",
        },
        output_file=f"{C.EventCrossings.SCREEN_TIME_IMAGES_DIR}/crossing_with_website_time.png"
    )
    plot_collisions_time_for_website(
        behs_with_labels={
            C.B_CHARTS_S: "S, Charts in slides",
            C.B_IMAGES_S: "S, Images in slides",
        },
        output_file=f"{C.EventCrossings.SCREEN_TIME_IMAGES_DIR}/website_crossing_time.png"
    )
