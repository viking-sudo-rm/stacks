import matplotlib.pyplot as plt
from matplotlib import animation, rc

from src.data.eval import EvalReader
from src.utils.trees import left_distances, right_distances


_MERGE = "merge"


class defaultlist(list):

    # Note: only supports positive indexing.

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise ValueError("defaultlist index needs to be integer.")

        if len(self) <= idx:
            self.extend(0 for _ in range(idx - len(self) + 1))

        return super().__getitem__(idx)


def normalize(dist):
    mass = sum(dist)
    return [x / mass for x in dist]


def main():
    reader = EvalReader()
    for length in range(10, 200, 10):
        dist = defaultlist()
        valid_path = "500:%d" % length

        for tree in reader.get_binary_trees(valid_path):
            for action in left_distances(tree):
                dist[action] += 1

        print(length, normalize(dist))


if __name__ == "__main__":
    main()
