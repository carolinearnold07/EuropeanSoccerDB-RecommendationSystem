import torch
from torch.utils.data import Dataset


class SoccerDataset(Dataset):
    def __init__(self, formations, positions, players, noise_std=0.0):
        """
        Args:
            formations: (N, 9, 2) tensor of [distance, angle]
            positions: (N, 2) tensor of [x, y]
            players: (N, 31) tensor of player attributes
            noise_std: Intensity of the augmentation
            shuffle_neighbors: Whether to randomize the order of the 9 teammates
        """
        self.formations = formations.float()
        self.positions = positions.float()
        self.players = players.float()
        self.noise_std = noise_std

    def __len__(self):
        return len(self.players)

    def __getitem__(self, idx):
        pos = self.positions[idx].clone()
        form = self.formations[idx].clone()
        player = self.players[idx].clone()

        # Noise Augmentation
        if self.noise_std > 0:
            # Jitter global position
            pos += torch.randn_like(pos) * self.noise_std

            # Jitter relative formation (dist and angle)
            form += torch.randn_like(form) * self.noise_std

        return pos, form, player
