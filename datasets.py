import torch
from torchvision import datasets, transforms

def load_data(dataset, batch_size, num_workers):
    if dataset == 'mnist':
        return load_MNIST( batch_size,  num_workers)
    elif dataset == 'celeba':
        return  load_CelebA( batch_size,  num_workers)
    elif dataset == 'cifar10':
        return  load_CIFAR10( batch_size,  num_workers)
    elif dataset == 'cifar100':
        return  load_CIFAR100( batch_size,  num_workers)

def load_MNIST(batch_size, num_workers, path='datasets/MNIST/'):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (1.0,))
    ])

    train_set = datasets.MNIST(path, train=True, transform=transform, download=True)
    test_set = datasets.MNIST(path, train=False, transform=transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                                num_workers=num_workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True,
                                                num_workers=num_workers, drop_last=True)

    return train_loader, test_loader

def load_CelebA(batch_size, num_workers, path='datasets/CelebA/'):
    # celeba_data = datasets.CelebA(path, download=True)
    # train_loader = torch.utils.data.DataLoader(celeba_data, batch_size=batch_size, shuffle=True, 
    #                                             split='train', num_workers=num_workers)
    # valid_loader = torch.utils.data.DataLoader(celeba_data, batch_size=batch_size, shuffle=True, 
    #                                           split='valid', num_workers=num_workers)
    # test_loader = torch.utils.data.DataLoader(celeba_data, batch_size=batch_size, shuffle=True, 
    #                                             split='test', num_workers=num_workers)

    train_set = datasets.CelebA(root=path, split='train', download=True)
    test_set = datasets.CelebA(root=path, split='test', download=True)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                shuffle=True, num_workers=num_workers,
                                                drop_last=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                shuffle=False, num_workers=num_workers,
                                                drop_last=True, pin_memory=True)

    return train_loader, test_loader
        

def load_CIFAR10(batch_size, num_workers, path='datasets/CIFAR10/'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = datasets.CIFAR10(root=path, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=path, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                shuffle=True, num_workers=num_workers,
                                                drop_last=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                shuffle=False, num_workers=num_workers,
                                                drop_last=True, pin_memory=True)

    return train_loader, test_loader

def load_CIFAR100(batch_size, num_workers, path='datasets/CIFAR100/'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = datasets.CIFAR100(root=path, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR100(root=path, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                shuffle=True, num_workers=num_workers,
                                                drop_last=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                shuffle=False, num_workers=num_workers,
                                                drop_last=True, pin_memory=True)

    return train_loader, test_loader

    