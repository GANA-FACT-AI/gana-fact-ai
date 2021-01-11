import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexFunctions import complex_relu, complex_max_pool2d
from modules import *

batch_size = 64
k = 8
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = datasets.MNIST('../data', train=True, transform=trans, download=True)
test_set = datasets.MNIST('../data', train=False, transform=trans, download=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size= batch_size, shuffle=True)
train_iterator = iter(train_loader)
test_loader = torch.utils.data.DataLoader(test_set, batch_size= batch_size, shuffle=True)

class ComplexNet(nn.Module):
    
    def __init__(self):
        super(ComplexNet, self).__init__()
        self.relu = InvariantReLU()
        self.bn   = InvariantBatchNorm()

        self.conv1 = ComplexConv2d(1, 20, 5, 1)
        self.conv2 = ComplexConv2d(20, 50, 5, 1)

        self.fc1 = ComplexLinear(4*4*50, 500)
        self.fc2 = ComplexLinear(500, 200)
        self.fc3 = nn.Linear(200, 10)
             
    def forward(self,xr,xi):
        # Encoder
        xr,xi = self.conv1(xr,xi)
        xr,xi = self.relu(xr,xi)
        xr,xi = complex_max_pool2d(xr,xi, 2, 2)
        a = xr
        self.discriminator = Discriminator(a.shape)
        
# rotation
        theta = torch.from_numpy(np.array(np.random.uniform(0, 2*math.pi)))
        rotated_r = torch.cos(theta)*xr - torch.sin(theta)*xi
        rotated_i = torch.sin(theta)*xr + torch.cos(theta)*xi

        # Processing Module
        
        xr,xi = self.bn(rotated_r,rotated_i)
        xr,xi = self.conv2(xr,xi)
        xr,xi = complex_relu(xr,xi)
        xr,xi = complex_max_pool2d(xr,xi, 2, 2)
        
        xr = xr.view(-1, 4*4*50)
        xi = xi.view(-1, 4*4*50)
        xr,xi = self.fc1(xr,xi)
        xr,xi = complex_relu(xr,xi)
        xr,xi = self.fc2(xr,xi)

        # Decoder
        x_orig_r = torch.cos(-theta)*xr - torch.sin(-theta)*xi

        x = self.fc3(x_orig_r)

        # Discriminator
        real_score = self.discriminator(a)
        fake_scores = []
        for i in range(k-1):
            theta = torch.from_numpy(np.array(np.random.uniform(0, 2*math.pi)))
            a_prime = torch.cos(-theta)*rotated_r - torch.sin(-theta)*rotated_i
            fake_scores.append(self.discriminator(a_prime))
        fake_scores = torch.stack(fake_scores, 0)

        # take the absolute value as output
        #x = torch.sqrt(torch.pow(xr,2)+torch.pow(xi,2))

        return F.log_softmax(x, dim=1), real_score, torch.mean(fake_scores, dim=0)
    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = ComplexNet().to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.00005)

def train(model, device, train_loader, train_iterator, optimizer, epoch):
    model.train()
    for batch_idx, (data_real, target) in enumerate(train_loader):
        data_real, target = data_real.to(device), target.to(device)
        data_fake, _ = train_iterator.next()
        data_fake = data_fake.to(device)
        optimizer.zero_grad()
        output, real_score, fake_score = model(data_real, data_fake)
        real_wasser_loss = torch.mean(real_score)
        fake_wasser_loss = torch.mean(fake_score * -1)
        gan_loss = real_wasser_loss - fake_wasser_loss

        loss = F.nll_loss(output, target)
        total_loss = gan_loss + loss
        total_loss.backward()
        model.discriminator._modules['score'].weight.data = model.discriminator._modules['score'].weight.data.clamp(-0.01,0.01)

        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {:3} [{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data_real), 
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader), 
                loss.item())
            )

# Run training on 50 epochs
for epoch in range(50):
    train_iterator = iter(train_loader)
    train(model, device, train_loader, train_iterator, optimizer, epoch)
