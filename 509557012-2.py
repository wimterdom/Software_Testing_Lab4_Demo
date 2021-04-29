import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import random,time
import matplotlib.pyplot as plt
num_epochs = 0

def cifar_loaders(batch_size, shuffle_test=False): 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10('./', train=True, download=True, 
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
    test = datasets.CIFAR10('./', train=False, 
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
        shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader

batch_size = 64
test_batch_size = 64
input_size = 3072

train_loader, _ = cifar_loaders(batch_size)
_, test_loader = cifar_loaders(test_batch_size)

class Conv7Net(nn.Module):
    def __init__(self):
        super(Conv7Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 84, kernel_size=5, stride=2, padding=1)

        self.conv2 = nn.Conv2d(84, 256, kernel_size=5, stride=2, padding=1)

        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=0)

        self.fc1 = nn.Linear(16*16*16,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.fc3 = nn.Linear(4096,10)
    
    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
                
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

net = Conv7Net()
print(net)
params = list(net.parameters())
print(len(params))
print(params[0].size())

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

ConvModel = Conv7Net()
ConvModel.to(device)
num_epochs = 100

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(ConvModel.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0002)



for epoch in range(num_epochs):
    avg_loss_epoch = 0
    batch_loss = 0
    total_batches = 0

    for i, (images, labels) in enumerate(train_loader):  
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)
        outputs = ConvModel(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   

        total_batches += 1     
        batch_loss += loss.item()

    avg_loss_epoch = batch_loss/total_batches
    print ('Epoch [{}/{}], Averge Loss:for epoch[{}, {:.4f}]' 
                   .format(epoch+1, num_epochs, epoch+1, avg_loss_epoch ))

# Test the Model
correct = 0.
total = 0.
predicted = 0.
for images, labels in test_loader:

    images = Variable(images).to(device)
    labels = Variable(labels).to(device)
    outputs_Conv_test = ConvModel(images)
    _, predicted = torch.max(outputs_Conv_test.data, 1)

    total += labels.size(0) 
    
    correct += (predicted == labels).sum().item()
    
print('Accuracy of the network on the 10000 test images: %d %%' % (     100 * correct / total))

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                   accuracy))
       