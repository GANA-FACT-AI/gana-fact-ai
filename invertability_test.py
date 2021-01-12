import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexFunctions import complex_relu, complex_max_pool2d

from torch.nn import Conv2d, Linear, BatchNorm1d, BatchNorm2d
from torch.nn.functional import relu, max_pool2d, dropout, dropout2d

from complexModules import *

batch_size = 64
k = 8
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = datasets.MNIST('../data', train=True, transform=trans, download=True)
test_set = datasets.MNIST('../data', train=False, transform=trans, download=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
train_iterator = iter(train_loader)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = True)


def rotate(xr,xi,theta):
    rotated_r = torch.cos(theta)*xr - torch.sin(theta)*xi
    rotated_i = torch.sin(theta)*xr + torch.cos(theta)*xi
    return rotated_r, rotated_i

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = Conv2d(1, 20, 5, 1, bias = True)
    def forward(self, x):
        x  = self.conv1(x)
        x = relu(x)
        x = max_pool2d(x, 2, 2)
        return x

class ProcessingModule(nn.Module):
    def __init__(self):
        super(ProcessingModule, self).__init__()
        self.relu = InvariantReLU()
        self.bn = InvariantBatchNorm()

        self.complex_conv2 = ComplexConv2d(20, 50, 5, 1, bias = False)   #kernel_size 5 -> 9 to keep size same when no maxpool
        self.fc1 = ComplexLinear(800, 500, bias  = False)
        self.fc2 = ComplexLinear(500, 200, bias  = False)

    def forward(self, xr, xi):
        xr,xi = self.bn(xr,xi)
        # print(xr,xi)
        xr,xi = self.complex_conv2(xr,xi)
        xr,xi = self.relu(xr,xi)

        xr,xi = invmaxpool2d(xr,xi, self.bn, kernel_size = 2, stride = 2)
        # print(xr.shape)
        xr = xr.view(-1,  xr.shape[1]*xr.shape[2]*xr.shape[3])
        # print(xr.shape)
        xi = xi.view(-1,  xi.shape[1]*xi.shape[2]*xi.shape[3])

        xr,xi = self.fc1(xr,xi)
        xr,xi = self.relu(xr,xi)
        xr,xi = self.fc2(xr,xi)

        return xr, xi

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(200, 10, bias  = True)

    def forward(self, x):
        x = self.fc3(x)
        return x


class ComplexNet(nn.Module):
    
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.encoder = Encoder()
        self.procmod = ProcessingModule()
        self.decoder = Decoder()

    def forward(self, x, theta):

        x = self.encoder(x)
        xr_init = x
        xi_init = torch.zeros(xr_init.shape, dtype = xr_init.dtype, device = xr_init.device)

        xr,xi = rotate(xr_init, xi_init, theta)
        xr,xi = self.procmod(xr,xi)
        x_orig, _  = rotate(xr, xi, - theta)

        xr_test, _ = self.procmod(xr_init,xi_init)
        delta = torch.mean(x_orig - xr_test)
#decoding
        x     = self.decoder(x_orig)
        return F.log_softmax(x, dim=1),  delta


    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = ComplexNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train(model, device, train_loader, optimizer, epoch):
    model.train()

    theta = torch.from_numpy(np.array(np.random.uniform(0, 2*math.pi)))
    print('Theta', theta)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output, _ = model(data, theta)

        with torch.no_grad():
            model.eval()
            _, delta = model(data, theta)

        model.train()

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {:3} [{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data), 
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader), 
                loss.item())
            )
            print('rotation', delta)
# Run training on 1 epochs
for epoch in range(1):

    model = train(model, device, train_loader, optimizer, epoch)
