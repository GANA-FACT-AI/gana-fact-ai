import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexFunctions import complex_relu, complex_max_pool2d

batch_size = 64
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = datasets.MNIST('../data', train=True, transform=trans, download=True)
test_set = datasets.MNIST('../data', train=False, transform=trans, download=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size= batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size= batch_size, shuffle=True)

class ComplexNet(nn.Module):
    
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.conv1 = ComplexConv2d(1, 20, 5, 1)
        self.bn  = ComplexBatchNorm2d(20)
        self.conv2 = ComplexConv2d(20, 50, 5, 1)
        self.fc1 = ComplexLinear(4*4*50, 500)
        self.fc2 = ComplexLinear(500, 200)
        self.fc3 = nn.Linear(200, 10)
             
    def forward(self,x):
        xr = x
        # imaginary part to zero
        xi = torch.zeros(xr.shape, dtype = xr.dtype, device = xr.device)
        '''
        z = torch.view_as_complex(torch.stack((xr,xi), 1))
        theta = 0.5 * math.pi
        i = torch.view_as_complex(torch.tensor([0, 1], dtype=torch.float32))
        rotated = torch.exp(theta*i) * z
        rotated_r = rotated.real
        rotated_i = rotated.imag
        '''
        xr,xi = self.conv1(xr,xi)
        xr,xi = complex_relu(xr,xi)
        xr,xi = complex_max_pool2d(xr,xi, 2, 2)
        
        theta = torch.from_numpy(np.array(math.pi))
        rotated_r = torch.cos(theta)*xr - torch.sin(theta)*xi
        rotated_i = torch.sin(theta)*xr + torch.cos(theta)*xi
        
        xr,xi = self.bn(rotated_r,rotated_i)
        xr,xi = self.conv2(xr,xi)
        xr,xi = complex_relu(xr,xi)
        xr,xi = complex_max_pool2d(xr,xi, 2, 2)
        
        xr = xr.view(-1, 4*4*50)
        xi = xi.view(-1, 4*4*50)
        xr,xi = self.fc1(xr,xi)
        xr,xi = complex_relu(xr,xi)
        xr,xi = self.fc2(xr,xi)

        x_orig_r = torch.cos(-theta)*xr - torch.sin(-theta)*xi

        x = self.fc3(x_orig_r)
        # take the absolute value as output
        #x = torch.sqrt(torch.pow(xr,2)+torch.pow(xi,2))

        return F.log_softmax(x, dim=1)
    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = ComplexNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
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

# Run training on 50 epochs
for epoch in range(50):


    train(model, device, train_loader, optimizer, epoch)