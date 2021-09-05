import itertools

list1 = [1, 2, 3]
metric1 = {
    "frames00": list1
}

list2 = [4, 5, 6]
metric2 = {
    "frames00": list2
}

[metric["frames00"] for metric in [metric1, metric2]]

elo = list(itertools.chain(*[metric["frames00"] for metric in [metric1, metric2]]))[0]
z = 1