import torch
import torchvision


def load_data(batch_size):
    celeba_data = torchvision.datasets.CelebA('datasets/CelebA')
    return torch.utils.data.DataLoader(celeba_data, batch_size=batch_size, shuffle=True, split='train'),\
           torch.utils.data.DataLoader(celeba_data, batch_size=batch_size, shuffle=True, split='valid'),\
           torch.utils.data.DataLoader(celeba_data, batch_size=batch_size, shuffle=True, split='test')