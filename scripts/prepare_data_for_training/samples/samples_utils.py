from scripts.utils.constants import C


def get_frame_ids_that_have_relevant_behs(frames_df, relevant_behs):
    ids = []
    for i, row in frames_df.iterrows():
        id = row[C.F_ID]
        beh_id = row[C.F_BEH_ID]

        if beh_id in relevant_behs:
            ids.append(id)
    return ids


def aggregate_dict_list_values(dictionary):
    values = set()
    for dlist in dictionary.values():
        for item in dlist:
            values.add(item)
    return values


def get_behaviors_to_labels(labels_to_behaviors):
    result = {}
    for label, behs in labels_to_behaviors.items():
        for beh in behs:
            result[beh] = label
    return result
