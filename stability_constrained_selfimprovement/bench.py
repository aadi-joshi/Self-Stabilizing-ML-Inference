#!/usr/bin/env python3
"""Minimal benchmark: measure per-iteration and per-epoch speed."""
import sys, time
sys.path.insert(0, '.')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.resnet import build_resnet
from utils.common import set_seed

set_seed(42)
device = torch.device('cpu')
model = build_resnet('resnet18_small', num_classes=2).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Params: {n_params:,}')

# Synthetic CIFAR-like data: 5000 samples
x = torch.randn(5000, 3, 32, 32)
y = torch.randint(0, 2, (5000,))
loader = DataLoader(TensorDataset(x, y), batch_size=128, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Warm up
for bx, by in loader:
    out = model(bx); loss = loss_fn(out, by); loss.backward(); optimizer.step(); optimizer.zero_grad()
    break

# Time 1 epoch
model.train()
t0 = time.time()
n_batches = 0
for bx, by in loader:
    out = model(bx)
    loss = loss_fn(out, by)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    n_batches += 1
elapsed = time.time() - t0
print(f'1 epoch ({n_batches} batches): {elapsed:.2f}s ({elapsed/n_batches*1000:.1f}ms/batch)')

# Time 5 epochs
t0 = time.time()
for ep in range(5):
    for bx, by in loader:
        out = model(bx)
        loss = loss_fn(out, by)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
elapsed = time.time() - t0
print(f'5 epochs: {elapsed:.2f}s')

# Eval timing
model.eval()
t0 = time.time()
with torch.no_grad():
    for bx, by in loader:
        model(bx)
elapsed = time.time() - t0
print(f'Eval 1 pass: {elapsed:.2f}s')

print(f'\nEstimate: 5 tasks × 5 epochs = ~{elapsed*5*5:.0f}s training + eval')
