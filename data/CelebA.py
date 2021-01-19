import torch
import torchvision
import torchvision.transforms as transforms


def load_data(batch_size, num_workers):
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    celeba_train_data = torchvision.datasets.CelebA('datasets/CelebA/', split='train', transform=transform, download=True)
    celeba_valid_data = torchvision.datasets.CelebA('datasets/CelebA/', split='valid', transform=transform, download=True)
    celeba_test_data = torchvision.datasets.CelebA('datasets/CelebA/', split='test', transform=transform, download=True)
    return torch.utils.data.DataLoader(celeba_train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True),\
           torch.utils.data.DataLoader(celeba_valid_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True),\
           torch.utils.data.DataLoader(celeba_test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)