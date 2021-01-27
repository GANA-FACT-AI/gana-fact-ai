from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms
#from sklearn.model_selection import train_test_split
import numpy as np

def load_data(dataset, batch_size, num_workers, adversary=False):
    if dataset == 'mnist':
        return load_MNIST(batch_size, num_workers, adversary)
    elif dataset == 'celeba':
        return  load_CelebA(batch_size, num_workers, adversary)
    elif dataset == 'cifar10':
        return  load_CIFAR10(batch_size, num_workers, adversary)
    elif dataset == 'cifar100':
        return  load_CIFAR100(batch_size, num_workers, adversary)

def split_dataset(dataset, split=0.7):
    split = int(len(dataset)*split)
    first_idx, second_idx = np.split(np.arange(len(dataset)), [split])
    return Subset(dataset, first_idx), Subset(dataset, second_idx)

def load_MNIST(batch_size, num_workers, adversary, path='datasets/MNIST/'):
    # Initilizes transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Downloads dataset
    train_set = datasets.MNIST(path, train=True, transform=transform, download=True)
    test_set = datasets.MNIST(path, train=False, transform=transform, download=True)

    # Splits dataset for privacymodel and adversary part
    train_privacy, train_adv = split_dataset(train_set)

    # Returns desired dataset split
    if not adversary:
        train_loader = DataLoader(train_privacy, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True, pin_memory=True)
    else:
        train_loader = DataLoader(train_adv, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=True, pin_memory=True)
    return train_loader, test_loader

def load_CelebA(batch_size, num_workers, adversary, path='datasets/CelebA/'):
    # Initilizes transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Downloads dataset
    train_set = datasets.CelebA(root=path, split='train', target_type='attr', download=True,
                                transform=transform)
    test_set = datasets.CelebA(root=path, split='test', target_type='attr', download=True,
                               transform=transform)

    # Splits dataset for privacymodel and adversary part
    train_privacy, train_adv = split_dataset(train_set)

    # Returns desired dataset split
    if not adversary:
        train_loader = DataLoader(train_privacy, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True, pin_memory=True)
    else:
        train_loader = DataLoader(train_adv, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=True, pin_memory=True)
    return train_loader, test_loader

def load_CIFAR10(batch_size, num_workers, adversary, path='datasets/CIFAR10/'):
    # Initilizes transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Downloads dataset
    train_set = datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=path, train=False, download=True, transform=transform)

    # Splits dataset for privacymodel and adversary part
    train_privacy, train_adv = split_dataset(train_set)

    # Returns desired dataset split
    if not adversary:
        train_loader = DataLoader(train_privacy, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True, pin_memory=True)
    else:
        train_loader = DataLoader(train_adv, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=True, pin_memory=True)
    return train_loader, test_loader

def load_CIFAR100(batch_size, num_workers, adversary, path='datasets/CIFAR100/'):
    # Initilizes transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Downloads dataset
    train_set = datasets.CIFAR100(root=path, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR100(root=path, train=False, download=True, transform=transform)

    # Splits dataset for privacymodel and adversary part
    train_privacy, train_adv = split_dataset(train_set)

    # Returns desired dataset split
    if not adversary:
        train_loader = DataLoader(train_privacy, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True, pin_memory=True)
    else:
        train_loader = DataLoader(train_adv, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=True, pin_memory=True)
    return train_loader, test_loader