import pandas as pd
import torch
from torch.utils.data import Dataset


class FailureData(Dataset):
    def __init__(self, file_name: str) -> None:
        super().__init__()
        raw = pd.read_csv(file_name)

        raw_x = raw.copy()
        raw_x = raw_x.drop('fault', axis=1)
        raw_x['steel_a300'] = raw_x['steel_a300'].astype('float32')
        raw_x['steel_a400'] = raw_x['steel_a400'].astype('float32')
        self.x_data = torch.tensor(raw_x.values.copy(), dtype=torch.float32)

        raw_y = raw.copy()
        raw_y['fault'] = raw_y['fault'].astype('category')
        self.raw_y_labels = torch.tensor(
            raw_y['fault'].cat.codes.values, dtype=torch.int64)
        self.one_hot_map_values = dict(
            enumerate(raw_y['fault'].cat.categories))
        self.y_data = torch.zeros(
            len(self.raw_y_labels), len(self.one_hot_map_values))
        self.y_data.scatter_(1, self.raw_y_labels.unsqueeze(1), 1.0)

    def match_encoding(self, output: torch.tensor):
        i = (output == 1).nonzero(as_tuple=True)[0].item()
        print("Index value: ", i)
        return self.one_hot_map_values.get(i)

    def get_encoding_labels(self):
        return self.one_hot_map_values

    def input_vector_length(self):
        return len(self.x_data[0])

    def output_vector_length(self):
        return len(self.y_data[0])

    def __getitem__(self, index: int):
        return self.x_data[index], self.raw_y_labels[index]

    def __len__(self) -> int:
        return len(self.y_data)
