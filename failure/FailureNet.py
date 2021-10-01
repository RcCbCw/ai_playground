from torch import nn


class FailureNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.li = nn.Linear(input_size, hidden_size)
        self.lh1 = nn.Linear(hidden_size, hidden_size)
        self.lh2 = nn.Linear(hidden_size, hidden_size)
        self.lo = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        activ_fn = nn.Tanh()
        x = self.li(x)
        x = activ_fn(x)
        x = self.lh1(x)
        x = activ_fn(x)
        x = self.lh2(x)
        x = activ_fn(x)
        x = self.lo(x)
        return x
