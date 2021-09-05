from typing import List

from scripts.prepare_data_for_training.samples.samples_v2 import SampleV2


class Analyzer:

    def __init__(self, samples_file):
        self.samples_file = samples_file

    def analyze(self):
        samples: List[SampleV2] = SampleV2.from_csv(self.samples_file)

        distribution = {
            0: 0,
            1: 0
        }
        for sample in samples:
            distribution[sample.label] = distribution[sample.label] + 1

        print(distribution)
