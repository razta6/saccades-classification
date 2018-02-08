import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F


class LeNet_binary(nn.Module):
    def __init__(self, LR=1e-3):
        super(LeNet_binary, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5,5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5,5))
        self.conv3 = nn.Conv2d(16, 32, (5,5))
        self.fc1   = nn.Linear(2592, 512)
        self.fc2   = nn.Linear(512, 128)
        self.fc3   = nn.Linear(128, 1)
    
    def forward(self, x):
        x = F.max_pool2d(x, (2,2))
        x = F.max_pool2d(x, (2,2))
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x

    def num_flat_features(self, x):
    	size = x.size()[1:]
    	num_features = 1
    	for s in size:
    		num_features *= s
    	return num_features


class LeNet_multiclass(nn.Module):
    def __init__(self, LR=1e-3, num_of_classes=3):
        super(LeNet_multiclass, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, (5,5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, (5,5))
        self.conv3 = nn.Conv2d(16, 32, (5,5))
        self.fc1   = nn.Linear(2592, 512)
        self.fc2   = nn.Linear(512, 128)
        self.fc3   = nn.Linear(128, num_of_classes)
    
    def forward(self, x):
        x = F.max_pool2d(x, (2,2))
        x = F.max_pool2d(x, (2,2))
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x

    def num_flat_features(self, x):
    	size = x.size()[1:]
    	num_features = 1
    	for s in size:
    		num_features *= s
    	return num_features

		

		



