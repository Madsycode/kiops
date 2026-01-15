import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv('/datasets/data.csv')
input_cols = ['snr_db', 'rsrp_dbm', 'rsrq_db', 'cqi', 'speed_mps', 'azimuth_deg', 'elevation_deg', 'beam_candidate']
target_cols = ['beam_index']

X = df[input_cols].values.astype(np.float32)
y = df[target_cols].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=32, shuffle=True)

model = nn.Sequential(
    nn.Linear(8, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    epoch_loss = 0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        outputs = model(bx)
        loss = criterion(outputs, by)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/10, Loss: {epoch_loss/len(train_loader):.4f}")

print("Training complete")

if not os.path.exists('/models'):
    os.makedirs('/models')

torch.save(model, '/models/beam-predictor-model_v1.pth')