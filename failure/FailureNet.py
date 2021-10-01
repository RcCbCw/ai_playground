from torch import nn


class FailureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.li = nn.Linear(27, 20)
        self.lh1 = nn.Linear(20, 20)
        self.lh2 = nn.Linear(20, 15)
        self.lh3 = nn.Linear(15, 13)
        self.lh4 = nn.Linear(13, 10)
        self.lh5 = nn.Linear(10, 10)
        self.lo = nn.Linear(10, 7)

    def forward(self, x):
        activ_fn = nn.Tanhshrink()
        x = self.li(x)
        x = activ_fn(x)

        x = self.lh1(x)
        x = activ_fn(x)

        x = self.lh2(x)
        x = activ_fn(x)

        x = self.lh3(x)
        x = activ_fn(x)

        x = self.lh4(x)
        x = activ_fn(x)

        x = self.lh5(x)
        x = activ_fn(x)

        x = self.lo(x)
        return x
