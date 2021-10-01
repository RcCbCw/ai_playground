from tqdm import tqdm
import torch
from torch import nn
from FailureData import FailureData
from FailureNet import FailureNet
from time import sleep


# Setup device
status = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(status)

# Training parameters
epochs = 10000
learning_rate = 0.00001

# Import data into a Dataset class
data = FailureData('data.csv')

# Create ANN
model = FailureNet()
model.to(device)
print(model)

# Split data into training and testing
data_len = len(data)
# print(data[0])
# print(model(data[0][0]))
training_len = int(data_len * 0.8)
test_len = data_len - training_len
train_data, test_data = torch.utils.data.random_split(
    data, [training_len, test_len], generator=torch.Generator().manual_seed(42))

# Dataloaders
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=500, shuffle=True, num_workers=6)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=100, shuffle=True, num_workers=6)

# Loss and optimization
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
try:
    with tqdm(range(epochs), unit=" epoch") as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            out_loss = None
            for X, y in train_loader:
                # Init setup
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(X)  # Compute prediction and loss
                loss = loss_fn(pred, y)  # Get losses
                # Backpropagation
                loss.backward()
                out_loss = loss
                optimizer.step()

            size = len(test_loader.dataset)
            num_batches = len(test_loader)
            test_loss, correct = 0, 0

            with torch.no_grad():
                for X, y in test_loader:
                    pred = model(X)
                    #pred = torch.reshape(pred, (1, len(pred)))
                    test_loss += loss_fn(pred, y).item()
                    correct += (pred.argmax(1) ==
                                y).type(torch.float).sum().item()

            test_loss /= num_batches
            correct /= size
            # Set bar
            tepoch.set_postfix(accuracy=f"{round(100.*correct, 3)}%",
                               AvgLoss=f"{round(test_loss, 4)}", loss=out_loss.item())
except KeyboardInterrupt:
    print("Training terminated early")
else:
    print("Done!")


print("----- Testing -----")
print(test_data[10][1])
out_pred = model(test_data[10][0])
print(out_pred)
print(data.match_encoding(out_pred))

# Save model
response = input("Save model (y/n): ")
if response == 'Y' or response == 'y':
    print("Saving model parameters")
else:
    print("Model parameters not saved")
