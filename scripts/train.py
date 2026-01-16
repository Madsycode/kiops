import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import os

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = pd.read_csv('/dataset/data.csv')
input_cols = ['snr_db', 'rsrp_dbm', 'rsrq_db', 'cqi', 'speed_mps', 'azimuth_deg', 'elevation_deg', 'beam_candidate']
target_cols = ['beam_index']

X = data[input_cols].values
y = data[target_cols].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

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
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch {epoch+1}/10, Loss: {epoch_loss/len(train_loader)}')

print('Training complete')
os.makedirs('/models', exist_ok=True)
torch.save(model, '/models/beam-predictor-model_v1.pth')