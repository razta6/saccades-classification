import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score


def split_train_test(X, Y, train_ratio):
	train_size = int(len(X) * train_ratio)
	X_train = X[:train_size]
	Y_train = Y[:train_size]
	X_test = X[train_size:]
	Y_test = Y[train_size:]

	return X_train, Y_train, X_test, Y_test


def train(X, Y, net, LR, epochs, TRAIN_RATIO, IMG_DIM):
    # arrange the data
    print('Train\Test split')
    X, Y = shuffle(X, Y)
    X_train, Y_train, X_test, Y_test = split_train_test(X, Y, TRAIN_RATIO)
    
    X_train = X_train.reshape(X_train.shape[0], 1, IMG_DIM, IMG_DIM)
    X_train = X_train.astype(float)
    #X_train /= 255.0
    
    X_test = X_test.reshape(X_test.shape[0], 1, IMG_DIM, IMG_DIM)
    X_test = X_test.astype(float)
    #X_test /= 255.0
    
    Y_train = Y_train.astype(int)

    
    # make torch variables
    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    Y_train = torch.from_numpy(Y_train)
    Y_train = Y_train.view(Y_train.shape[0], -1)
    print(X_train.size(), Y_train.size())
    print() 
    
    print('Training...')

    # training parameters
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)

    threshold = np.full((len(Y_test)), 0.5)
    train_size = X_train.shape[0]
    index = 0
    batch_size = 32
        
    
    # training phase
    for epoch in range(epochs):
        if index + batch_size >= train_size:
            index = 0
        else:
            index = index + batch_size
    
        mini_data = Variable(X_train[index:(index+batch_size)].clone())
        mini_label = Variable(Y_train[index:(index+batch_size)].clone(), requires_grad=False)
        mini_data = mini_data.type(torch.FloatTensor)
        mini_label = mini_label.type(torch.FloatTensor)
    
        optimizer.zero_grad()
        mini_out = net(mini_data)
        mini_label = mini_label.view(mini_label.size())
        mini_loss = criterion(mini_out, mini_label)
        mini_loss.backward()
        optimizer.step()
    
        if (epoch+1) % 10 == 0:
            test_data = Variable(X_test.clone())
            test_data = test_data.type(torch.FloatTensor)
            out = net(test_data)
            out_np = np.concatenate(out.data.numpy())
            y_pred =  out_np > threshold
    
            acc = accuracy_score(Y_test, y_pred)
    
            print('Epoch [%d/%d] Train Loss: %f, Test Accuracy: %f'
                %(epoch+1, epochs, mini_loss.data[0], acc))

    print()
    print('Done!')