import torch
from torch import nn


class FFN(nn.Module):
    def __init__(self):
        super(FFN, self).__init__()
        hidden_size = 30
        self.li = nn.Linear(1, hidden_size)
        self.h1 = nn.Linear(hidden_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, 10)
        #self.h3 = nn.Linear(hidden_size, 10)
        self.lo = nn.Linear(10, 2)

    def forward(self, x):
        m = nn.Sigmoid()
        x = self.li(x)
        x = m(x)
        x = self.h1(x)
        x = m(x)
        x = self.h2(x)
        x = self.lo(x)
        return x
