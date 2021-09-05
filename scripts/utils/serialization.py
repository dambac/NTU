import json as j
from pathlib import Path


def save_json(json_dict, output):
    dir_name = Path(output).parent
    Path(dir_name).mkdir(exist_ok=True, parents=True)

    with open(output, 'w') as file:
        j.dump(json_dict, file, indent=2)


def read_json(input_file):
    with open(input_file, 'r') as file:
        return j.load(file)


def to_json_dict(obj, overrides=None):
    json_dict = {}

    for field, field_value in vars(obj).items():
        if overrides and field in overrides:
            json_dict[field] = overrides[field]
        else:
            json_dict[field] = field_value

    return json_dict


def from_json_dict(json_dict, clazz, overrides=None):
    obj = clazz.empty()

    for field, value in json_dict.items():
        if overrides and overrides.get(field):
            setattr(obj, field, overrides[field](value))
        else:
            setattr(obj, field, value)

    return obj


def defaults(obj, defaults_provider):
    dic = defaults_provider(obj)

    for field, value in vars(obj).items():
        if field in dic and value is None:
            setattr(obj, field, dic[field])

    return obj


def list_to_str(list):
    joined = ",".join([str(item) for item in list])
    return f"[{joined}]"


def list_from_str(value, string_values=False):
    value = value[1:-1]
    if value == "":
        return []
    if string_values:
        return [item for item in value.split(",")]
    return [eval(item) for item in value.split(",")]
