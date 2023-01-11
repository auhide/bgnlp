from typing import List

from torch.utils.data import Dataset, random_split


def iterate_corpus(tokens: List[str], batch_size: int):
    for i in range(0, len(tokens) - batch_size):
        yield tokens[i:i + batch_size]


def train_validation_split(dataset: Dataset, train_size=0.8):
    train_set_size = int(len(dataset) * train_size)
    valid_set_size = len(dataset) - train_set_size
    datasets_lengths = [train_set_size, valid_set_size]

    # Splitting the input dataset into training and validation set.
    train_dataset, valid_dataset = random_split(dataset, datasets_lengths)

    return train_dataset, valid_dataset