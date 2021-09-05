"""
Prepare a plot about the amount of time our behaviors are indeed visible on the screen according to layout
type classifiers prepared in previous works.
"""
import matplotlib.pyplot as plt

from scripts.dataset_analysis.screen_time_for_events.calculate_events_screen_time import read_events_screen_time
from scripts.utils.common import mkdir_path
from scripts.utils.constants import C


def plot_events_screen_time(behs_with_labels, output_file):
    mkdir_path(output_file)

    # load already prepared data
    events_screen_time = read_events_screen_time("events_screen_time.pickle")

    fig, ax = plt.subplots()

    for index, (beh, label) in enumerate(behs_with_labels.items()):
        beh_screen_time = events_screen_time[beh]
        beh_screen_time = [t * 100 for t in beh_screen_time]
        ax.boxplot(beh_screen_time, positions=[index])

    ax.set_xticklabels(behs_with_labels.values())
    plt.ylabel('Part of event annotated with content type "Screen" (%)', fontsize=10)
    plt.savefig(output_file)
    # plt.show()


if __name__ == "__main__":
    plot_events_screen_time(
        behs_with_labels={
            C.B_WEBSITE_S: "S, Website",
            C.B_CHARTS_S: "S, Charts in slides",
            C.B_IMAGES_S: "S, Images in slides",
        },
        output_file=f"{C.EventCrossings.SCREEN_TIME_IMAGES_DIR}/screen_time.png"
    )
