import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)

X = torch.randn(1000, 2)
y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=30, shuffle=True)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(2, 10)
        self.relu = nn.ReLU()
        self.out = nn.Linear(10, 1)
        self.sigm = nn.Sigmoid()
    
    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = self.out(x)
        x = self.sigm(x)
        return x
    
 
model = SimpleNet()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100

for i in range(epochs):
    epoch_loss = 0
    for X, y in dataloader:
        # forward
        out = model(X)
        loss = criterion(out, y)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    print(f"Ep {i} loss: {epoch_loss}")



