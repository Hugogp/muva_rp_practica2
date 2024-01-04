

_label_to_index_dict = {'forest': 0, 'fungus': 1, 'grass': 2, 'leaves': 3, 'salad': 4}
_index_to_label_dict = {0: 'forest', 1: 'fungus', 2: 'grass', 3: 'leaves', 4: 'salad'}


def label_to_index(label: str) -> int:
    return _label_to_index_dict[label]


def index_to_label(index: int) -> str:
    return _index_to_label_dict[index]


def get_labels_distribution(labels: [str]) -> dict:
    labels_dist = {'forest': 0, 'fungus': 0, 'grass': 0, 'leaves': 0, 'salad': 0}

    for label in labels:
        labels_dist[label] += 1

    return labels_dist