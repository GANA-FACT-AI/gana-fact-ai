import torch
import torchvision


def load_data(batch_size, num_workers):
    celeba_data = torchvision.datasets.CelebA('datasets/CelebA/', download=True)
    return torch.utils.data.DataLoader(celeba_data, batch_size=batch_size, shuffle=True, split='train', num_workers=num_workers),\
           torch.utils.data.DataLoader(celeba_data, batch_size=batch_size, shuffle=True, split='valid', num_workers=num_workers),\
           torch.utils.data.DataLoader(celeba_data, batch_size=batch_size, shuffle=True, split='test', num_workers=num_workers)