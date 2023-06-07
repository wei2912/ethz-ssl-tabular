from collections import Counter
import random


def get_balanced_ids(dataset, seed=0):
    random.seed(seed)

    class_nums = Counter((label for (_, label) in dataset))
    _, class_num = class_nums.most_common()[-1]

    ids_labels = [(i, label) for i, (_, label) in enumerate(dataset)]
    random.shuffle(ids_labels)

    class_nums = Counter()
    for i, label in ids_labels:
        if class_nums[label] < class_num:
            class_nums.update([label])
            yield i
