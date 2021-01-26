import torch
import torchvision
import torchvision.transforms as transforms


def load_data(batch_size, num_workers, path='datasets/CIFAR10/'):
    transform = transforms.Compose(
    [transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root=path, train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers, drop_last=True, pin_memory=True)
    return trainloader, testloader
