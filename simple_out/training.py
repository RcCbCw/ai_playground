import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, dataloader
from TestDataSet import NumpyData
from FFN import FFN

# Create data
train_x = np.arange(0, 100, 0.5)
length = len(train_x)
train_x = np.reshape(train_x, (length, 1))

linear_y = train_x.copy() + 2
expo_y = np.array([x**2 for x in train_x])
train_y = np.zeros((length, 2))
for i in range(length):
    train_y[i][0] = linear_y[i]
    train_y[i][1] = expo_y[i]
train_dataset = NumpyData(train_x, train_y)


test_x = np.random.rand(30)*50
test_x = np.reshape(test_x, (30, 1))
length_test = len(test_x)

linear_test_y = test_x.copy() + 2
expo_test_y = np.array([x**2 for x in test_x])
test_y = np.zeros((length_test, 2))
for i in range(length_test):
    test_y[i][0] = linear_test_y[i]
    test_y[i][1] = expo_test_y[i]
test_dataset = NumpyData(test_x, test_y)

model = FFN()

train_dataloader = DataLoader(train_dataset, batch_size=200)
test_dataloader = DataLoader(test_dataset, batch_size=30)


learning_rate = 0.001
epochs = 50000

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

in_data = train_dataloader.dataset[0][0]


def train_loop(dataloader, model, loss_fn, optimizer, pr):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0 and pr:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, pr):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            diff = pred - y
            diff = diff.abs()
            correct_ans = torch.zeros(len(y))
            for i, (i0, i1) in enumerate(diff):
                correct_ans[i] = 1 if i0 < 1 and i1 < 3 else 0
            correct = correct_ans.sum()

    test_loss /= num_batches
    correct /= size
    if pr:
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(epochs):
    pr = False
    if t % 500 == 0:
        print(f"Epoch {t+1}\n-------------------------------")
        pr = True
    train_loop(train_dataloader, model, loss_fn, optimizer, pr)
    if pr:
        test_loop(test_dataloader, model, loss_fn, pr)
print("Done!")

test_output = model(torch.tensor([5.]))
print(train_dataset[10])
print(test_output)
