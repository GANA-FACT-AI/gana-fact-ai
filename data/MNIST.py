import torch
from torchvision import datasets, transforms


def load_data(batch_size):
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = datasets.MNIST('../data', train=True, transform=trans, download=True)
    test_set = datasets.MNIST('../data', train=False, transform=trans, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size= batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size= batch_size, shuffle=True)
    return train_loader, test_loader