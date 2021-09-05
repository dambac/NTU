"""
Prepare heatmap describing how often certain behaviors collide with each other. Calculations are based on state events.
"""
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from scripts.dataset_analysis.overlapping_of_events.event_crossings import EventWithCrossings
from scripts.utils.common import pickle_obj, unpickle_obj
from scripts.utils.constants import C


def prepare_data(behs_with_labels):
    """
    Prepares data for heatmap visualization. This is only a little optimization in order not to calculate date
    time each time we want to change a little thing about our plot.

    :param behs_with_labels: labels for behaviors that we want to analyze
    """
    # load already prepared data about events crossings
    crossings: List[EventWithCrossings] = EventWithCrossings.load("new_crossings.pickle")
    behs_of_interest = list(behs_with_labels.keys())

    data = []
    for beh in behs_of_interest:
        beh_data = []
        beh_crossings = [cross for cross in crossings if cross.beh_id == beh]
        occurrences_of_beh = len(beh_crossings)

        for beh2 in behs_of_interest:
            if beh2 == beh:
                beh_data.append(float('nan'))
                continue

            number_of_crosses = 0
            for cross in beh_crossings:
                cross_events = cross.crossing_events

                for ce in cross_events:
                    if ce.beh_id == beh2:
                        number_of_crosses += 1
                        break

            beh_data.append(number_of_crosses / occurrences_of_beh)
        data.append(beh_data)

    pickle_obj(data, f"{C.EventCrossings.EVENT_LAYOUT_DISTRIBUTION_DIR}/heatmap.pickle")


def plot_state_crossings(behs_with_labels, output_file):
    data = unpickle_obj(f"{C.EventCrossings.EVENT_LAYOUT_DISTRIBUTION_DIR}/heatmap.pickle")

    # artificially add more color (this might not be needed)
    data_pumped = []
    for row in data:
        new_row = [(item + 0.2) for item in row]
        data_pumped.append(new_row)

    behs_of_interest = list(behs_with_labels.keys())
    beh_labels = list(behs_with_labels.values())

    fig, ax = plt.subplots()

    ax.imshow(data_pumped, cmap="Blues")

    ax.set_xticks(np.arange(len(behs_of_interest)))
    ax.set_yticks(np.arange(len(behs_of_interest)))

    ax.set_xticklabels(beh_labels)
    ax.set_yticklabels(beh_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(behs_of_interest)):
        for j in range(len(behs_of_interest)):
            number = data[i][j]
            if not np.math.isnan(number):
                text = "{:.2f}".format(number)
            else:
                text = "-"
            ax.text(j, i, text,
                    ha="center", va="center", color="black", fontsize=16, alpha=1)

    fig.tight_layout()
    ax.tick_params(axis="both", which="major", labelsize=12)

    plt.savefig(output_file)
    # plt.show()


if __name__ == "__main__":
    behs_with_labels = {
        C.B_CHARTS_S: "S, Charts in slides",
        C.B_IMAGES_S: "S, Images in slides",
        C.B_WEBSITE_S: "S, Website"
    }
    # uncomment and run it
    # prepare_data(behs_with_labels)
    plot_state_crossings(
        behs_with_labels=behs_with_labels,
        output_file=f"{C.EventCrossings.SCREEN_TIME_IMAGES_DIR}/state_collisions_heatmap.png"
    )
