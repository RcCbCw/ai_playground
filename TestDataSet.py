import numpy as np
import torch
from torch.utils.data import Dataset


class NumpyData(Dataset):
    def __init__(self, x_input, y_output):
        super().__init__()
        self.x = torch.tensor(x_input.copy(), dtype=torch.float32)
        self.y = torch.tensor(y_output.copy(), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]
