#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sqlite3
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn


# # Load Data

# In[2]:


db_path = Path("../../data/database.sqlite")

conn = sqlite3.connect(db_path.as_posix())

df = pd.read_sql_query("SELECT * FROM Player_Formation_Preprocessed", conn)

conn.close()


# In[3]:


formation_cols = (
    ["x", "y"]
    + [f"distance_{i + 1}" for i in range(9)]
    + [f"angle_{i + 1}" for i in range(9)]
)
player_stats_cols = [
    "preferred_foot",
    "attacking_work_rate",
    "defensive_work_rate",
    "crossing",
    "finishing",
    "heading_accuracy",
    "short_passing",
    "volleys",
    "dribbling",
    "curve",
    "free_kick_accuracy",
    "long_passing",
    "ball_control",
    "acceleration",
    "sprint_speed",
    "agility",
    "reactions",
    "balance",
    "shot_power",
    "jumping",
    "stamina",
    "strength",
    "long_shots",
    "aggression",
    "interceptions",
    "positioning",
    "vision",
    "penalties",
    "marking",
    "standing_tackle",
    "sliding_tackle",
]
df_formation = df[formation_cols]
df_player = df[player_stats_cols]


# # Transform to Tensor

# In[4]:


df_formation.head()


# In[5]:


formation_numpy = df_formation.to_numpy()


# In[6]:


positions = torch.tensor(formation_numpy[:, :2], dtype=torch.float32)
positions.shape


# In[7]:


formation_relpos = torch.tensor(formation_numpy[:, 2:], dtype=torch.float32)

distance_indices = list(range(0, 9))
angle_indices = list(range(9, 18))

formation_distances = formation_relpos[:, distance_indices]
angle_indices = formation_relpos[:, angle_indices]

formations = torch.stack([formation_distances, angle_indices], dim=2)
formations.shape


# In[8]:


players = torch.tensor(df_player.to_numpy(), dtype=torch.float32)
players.shape


# # Build Two-Tower Neural Network

# ### Formation Tower

# In[57]:


class FormationTower(nn.Module):
    def __init__(self):
        super().__init__()
        # Phi: (B, 9, 2) -> (B, 9, 64)
        self.phi = nn.Sequential(
            nn.Linear(in_features=2, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.LeakyReLU(),
        )

        # Rho: (B, 64 + 2) -> (B, 64)
        # We match in_features to the phi output (64) + position (2)
        self.rho = nn.Sequential(
            nn.Linear(in_features=64 + 2, out_features=128),
            nn.BatchNorm1d(128),  # Must match out_features above
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=64),  # Final embedding dim: 64
        )

    def forward(self, formation, position):
        x = self.phi(formation)  # (Batch, 9, 64)
        x = torch.sum(x, dim=1)  # (Batch, 64)

        x = torch.concat([x, position], dim=1)  # (Batch, 66)
        return self.rho(x)


class PlayerTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=31, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=64),  # Final embedding dim: 64
        )

    def forward(self, player):
        return self.mlp(player)


# ### Two-Tower Head

# In[58]:


class TwoTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.formation_tower = FormationTower()
        self.player_tower = PlayerTower()

    def forward(self, formation, position, player):
        formation_embedding = self.formation_tower(formation, position)
        player_embedding = self.player_tower(player)

        f_norm = formation_embedding / (
            torch.linalg.norm(formation_embedding, dim=1, keepdim=True) + 1e-8
        )
        p_norm = player_embedding / (
            torch.linalg.norm(player_embedding, dim=1, keepdim=True) + 1e-8
        )

        similarity_score = (f_norm * p_norm).sum(dim=1)

        return f_norm, p_norm, similarity_score


# # Dataset

# In[59]:


import numpy as np
from sklearn.model_selection import train_test_split
from dataset import SoccerDataset


# In[60]:


indices = np.arange(len(players))
train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)


# In[61]:


train_dataset = SoccerDataset(
    formations[train_idx], positions[train_idx], players[train_idx], noise_std=0.01
)

val_dataset = SoccerDataset(
    formations[val_idx], positions[val_idx], players[val_idx], noise_std=0.0
)

test_dataset = SoccerDataset(
    formations[test_idx], positions[test_idx], players[test_idx], noise_std=0.0
)


# In[62]:


train_dataset[0]


# # Data Loader

# In[63]:


from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=4)


# # Train Loop

# In[71]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[72]:


def train_one_epoch(loader, model, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for pos, form, player in loader:
        pos, form, player = pos.to(device), form.to(device), player.to(device)

        f_embed, p_embed, _ = model(form, pos, player)

        temp = 0.1
        logits = torch.matmul(f_embed, p_embed.T) / temp

        batch_size = f_embed.size(0)
        labels = torch.arange(batch_size).to(device)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# In[73]:


def validate(loader, model, criterion):
    model.eval()  # Set to evaluation mode (disables dropout)
    total_loss = 0.0

    with torch.no_grad():  # Disable gradient calculation to save memory/speed
        for pos, form, player in loader:
            pos, form, player = pos.to(device), form.to(device), player.to(device)

            f_embed, p_embed, _ = model(form, pos, player)

            temp = 0.1
            logits = torch.matmul(f_embed, p_embed.T) / temp
            labels = torch.arange(f_embed.size(0)).to(device)

            loss = criterion(logits, labels)
            total_loss += loss.item()

    return total_loss / len(loader)


# In[74]:


# Hyperparameters
num_epochs = 150
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, Optimizer, and Loss
model = TwoTower().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Lists to track progress
train_losses = []
val_losses = []


# In[75]:


print(f"Starting training on {device}...")

for epoch in range(num_epochs):
    # --- TRAIN ---
    train_loss = train_one_epoch(train_loader, model, optimizer, criterion)

    # --- VALIDATE ---
    val_loss = validate(val_loader, model, criterion)

    # Track metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
    )

    # --- OPTIONAL: SAVE BEST MODEL ---
    if val_loss == min(val_losses):
        torch.save(model.state_dict(), "best_soccer_model.pth")
        print("--> Model saved!")


# In[76]:


def evaluate_top_k(model, loader, device, k=10):
    model.eval()
    all_f_embeds = []
    all_p_embeds = []

    print("Extracting test embeddings...")
    with torch.no_grad():
        for pos, form, player in loader:
            pos, form, player = pos.to(device), form.to(device), player.to(device)
            f_norm, p_norm, _ = model(form, pos, player)

            all_f_embeds.append(f_norm.cpu())
            all_p_embeds.append(p_norm.cpu())

    # Concatenate all embeddings into two large tensors
    f_embeds = torch.cat(all_f_embeds)  # (N, 32)
    p_embeds = torch.cat(all_p_embeds)  # (N, 32)

    num_samples = f_embeds.shape[0]
    top_1_correct = 0
    top_k_correct = 0

    print(f"Calculating Top-{k} accuracy for {num_samples} samples...")

    # Process in chunks to avoid memory issues
    chunk_size = 2000
    for i in range(0, num_samples, chunk_size):
        end = min(i + chunk_size, num_samples)

        # Similarity matrix for this chunk against ALL players
        # Shape: (chunk_size, total_test_samples)
        logits = torch.matmul(f_embeds[i:end], p_embeds.T)

        # The correct player for formation 'j' is at index 'j'
        targets = torch.arange(i, end).view(-1, 1)

        # Get Top-K indices
        _, top_indices = torch.topk(logits, k=k, dim=1)

        # Check if the target index is in the top 1
        top_1_correct += (top_indices[:, :1] == targets).sum().item()

        # Check if the target index is anywhere in the top K
        top_k_correct += (top_indices == targets).any(dim=1).sum().item()

    top1_acc = (top_1_correct / num_samples) * 100
    topk_acc = (top_k_correct / num_samples) * 100

    print(f"\n--- Results ---")
    print(f"Top-1 Accuracy:  {top1_acc:.2f}%")
    print(f"Top-{k} Accuracy: {topk_acc:.2f}%")

    return top1_acc, topk_acc


# In[77]:


test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TwoTower().to(device)
model_path = "best_soccer_model.pth"
state_dict = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()

evaluate_top_k(model, test_loader, device, k=10)

