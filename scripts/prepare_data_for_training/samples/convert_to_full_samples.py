from tqdm import tqdm

from scripts.prepare_data_for_training.samples.samples_v2 import *


def samples_to_full_samples(samples: List[SampleV2], transfer_views_dir):
    # result
    full_samples = []
    full_samples_ids = []

    sample: SampleV2
    for sample in tqdm(samples):

        split_names = sample.splits_names
        view_name1 = split_names[0]
        view_name2 = split_names[0]

        view1 = torch.load(f"{transfer_views_dir}/{view_name1}.pt")[0]
        if view_name2 == view_name1:
            view2 = view1
        else:
            view2 = torch.load(f"{transfer_views_dir}/{view_name2}.pt")[0]

        views = torch.stack([view1, view2])
        label = torch.tensor(sample.label)

        full_samples.append(FullSampleV2(frame_id=sample.id,
                                         split_names=sample.splits_names,
                                         views=views,
                                         label=label))
        full_samples_ids.append(sample.id)

    return full_samples, full_samples_ids
