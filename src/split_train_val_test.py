import numpy as np


def split_train_val_test(arr, train_frac=4 / 6, test_frac=5 / 6):
    """
    Splits a list of values into a number of chunks based on two cutting points.

    :param arr: list of values (can be a np.ndarray, a regular list, or a pd.DataFrame)
    :param train_frac: fraction of the index to cut at for train data
    :param test_frac: fraction of the index to cut at for validation data
    :return: tuple (train, val, test)
    """
    return np.split(arr, [int(train_frac * len(arr)), int(test_frac * len(arr))])


if __name__ == "__main__":
    with open("dataset/reviews.txt", encoding="utf8") as f:
        lines = f.readlines()

    dataset = split_train_val_test(lines)

    for values, output_filename in zip(dataset, ["train", "dev", "test"]):
        with open(f"dataset/{output_filename}.txt", "w", encoding="utf8") as f:
            f.writelines(values)
