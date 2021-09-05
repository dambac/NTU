from pathlib import Path
from typing import List

import pandas as pd
import torch

from scripts.utils.constants import C

ID_FORMAT = "{video}__{time}__{event_id}__{beh}__{ra}"


class SampleV2:

    def __init__(self, frame_id, splits_names, label):
        self.id = frame_id
        self.splits_names = splits_names
        self.label = label

    def extract_video_name(self):
        return self.id.split("__")[0]

    @staticmethod
    def to_csv(samples, name):
        df_data = []

        sample: SampleV2
        for sample in samples:
            df_data.append([
                sample.id,
                sample.splits_names,
                sample.label
            ])

        df = pd.DataFrame(data=df_data,
                          columns=[C.S_FRAME_ID, C.S_SPLITS_NAMES, C.S_LABEL])

        output_dir = C.SAMPLES_DATAFRAMES_DIR
        Path(output_dir).mkdir(exist_ok=True)

        output_file = f"{output_dir}/{name}.csv"
        df.to_csv(output_file, index=False)

    @staticmethod
    def from_csv(name):
        input_file = f"{C.SAMPLES_DATAFRAMES_DIR}/{name}.csv"
        df = pd.read_csv(input_file, converters=C.S_CONVERTERS)

        samples = []
        for _, row in df.iterrows():
            samples.append(SampleV2(
                frame_id=row[C.S_FRAME_ID],
                splits_names=row[C.S_SPLITS_NAMES],
                label=row[C.S_LABEL]
            ))
        return samples

    def __repr__(self):
        return f"{self.id, self.label, self.splits_names}"


class FullSampleV2:
    def __init__(self, frame_id, split_names, label, views):
        self.id = frame_id
        self.split_names = split_names
        # torch
        self.views = views
        # torch
        self.label = label


class SamplesDatasetV2(torch.utils.data.Dataset):

    def __init__(self, samples: List[FullSampleV2], allowed_ids=None):
        super(SamplesDatasetV2, self).__init__()

        self.index_to_id = {}
        self.items_dict = {}
        self.length = 0

        current_index = 0

        for index, sample in enumerate(samples):
            if allowed_ids is None or sample.id in allowed_ids:
                self.index_to_id[current_index] = sample.id
                self.items_dict[sample.id] = sample

                current_index = current_index + 1
                self.length = self.length + 1

    def get_all_ids(self):
        return list(self.index_to_id.values())

    def get_by_id(self, id):
        return self.items_dict[id]

    def __len__(self):
        return len(self.items_dict)

    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError

        id = self.index_to_id[idx]
        sample: FullSampleV2 = self.items_dict[id]
        return sample.views, sample.label, sample.id
