import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
import os


transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(0.1307,0.3087)])

train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data',train=True,transform=transform,download=True),
                                           batch_size=32,shuffle=True)

test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data',train=False,transform=transform,download=True),
                                           batch_size=32,shuffle=True)

class Dense(nn.Module):
    
    def __init__(self):
        super(Dense,self).__init__()
        self.layer1 = nn.Linear(28*28,64)
        self.layer2 = nn.Linear(64,10)
        self.init_weights()#autre initialisation que celle de base
        
    def init_weights(self):
        for p in self.parameters():
            if p.ndim == 1:
                nn.init.normal_(p) #initialisation selon loi normale centrée réduite
            else:
                nn.init.kaiming_normal_(p)
            
    def forward(self,x):
        
        x = x.view(-1,28*28)
        x = F.relu(self.layer1(x))
        x = F.softmax(self.layer2(x),dim=-1)
        return x



class CNN(nn.Module):
    
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.dense1 = nn.Linear(20*4*4,50)
        self.dense2 = nn.Linear(50,10)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.ndim == 1:
                nn.init.normal_(p) #initialisation selon loi normale centrée réduite
            else:
                nn.init.kaiming_normal_(p)
            
    def forward(self,x):
        
        x = self.conv1(x)
        x = F.max_pool2d(x,2)
        # x = F.dropout2d(x,0.5,training=self.training)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.max_pool2d(x,2)
        x = F.dropout2d(x,0.5,training=self.training)
        x = F.relu(x)
        
        x = x.view(-1,320)
        x = F.relu(F.dropout(self.dense1(x),0.5,training=self.training))
        x = F.softmax(self.dense2(x),dim=-1)
        return x



def model_dense(N_EPOCH=1):

    model = Dense()
    global_loss = []
    global_acc = []

    lr = 1e-3
    model_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    def accuracy(proba,label):
        correct = 0
        batch_size = label.size(0) #equivalent à len
        prediction = torch.argmax(proba,dim=-1)
        for i, pred in enumerate(prediction):
            if pred == label[i]:
                correct += 1
        correct /= batch_size
        return correct


    for n_epoch in range(N_EPOCH):
        print(f"epoch {n_epoch+1}/{N_EPOCH}")
        for batch, (data_batch, label_batch) in enumerate(train_loader):
            
            optimizer.zero_grad()#on remet à 0 les gradients de toutes les variables
            pred_batch = model(data_batch)
            loss = model_loss(pred_batch,label_batch)
            loss.backward()
            optimizer.step()
            acc = accuracy(pred_batch,label_batch)
            
            print(f"\r batch {batch}/{int(60000/32)-1} loss = {loss : .3} acc = {acc : .3}",end='') 
            
        print()
        loss = 0.
        acc = 0.
        for batch, (data_batch, label_batch) in enumerate(test_loader):
            with torch.no_grad():#on ne suit pas les gradients 
                pred_batch = model(data_batch)
                loss += model_loss(pred_batch,label_batch)
                acc += accuracy(pred_batch,label_batch)
        loss /= batch
        acc /= batch
        global_loss.append(loss)
        global_acc.append(acc)
        print(f" validation : loss = {loss: .3} ; acc = {acc : .3}")
    return global_loss,global_acc
        
def model_cnn(N_EPOCH=1):
    
    global_loss = []
    global_acc = []
    model = CNN()

    lr = 1e-3
    model_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)

    def accuracy(proba,label):
        correct = 0
        batch_size = label.size(0) #equivalent à len
        prediction = torch.argmax(proba,dim=-1)
        for i, pred in enumerate(prediction):
            if pred == label[i]:
                correct += 1
        correct /= batch_size
        return correct


    for n_epoch in range(N_EPOCH):
        print(f"epoch {n_epoch+1}/{N_EPOCH}")
        model.training = True
        for batch, (data_batch, label_batch) in enumerate(train_loader):
            
            optimizer.zero_grad()#on remet à 0 les gradients de toutes les variables
            pred_batch = model(data_batch)
            loss = model_loss(pred_batch,label_batch)
            loss.backward()
            optimizer.step()
            acc = accuracy(pred_batch,label_batch)
            
            print(f"\r batch {batch}/{int(60000/32)-1} loss = {loss : .3} acc = {acc : .3}",end='') 
            
        print()
        loss = 0.
        acc = 0.
        model.training = False
        for batch, (data_batch, label_batch) in enumerate(test_loader):
            with torch.no_grad():#on ne suit pas les gradients 
                pred_batch = model(data_batch)
                loss += model_loss(pred_batch,label_batch)
                acc += accuracy(pred_batch,label_batch)
        loss /= batch
        acc /= batch
        global_loss.append(loss)
        global_acc.append(acc)
        print(f" validation : loss = {loss: .3} ; acc = {acc : .3}")
    return global_loss,global_acc
        
def plot_result(x,y_axis1,y_axis2,title):
    x = np.arange(x)
    fig = plt.figure()
    ax = plt.gca()
    plt.title(title)
    plt.plot(x,y_axis1,label="Dense")
    plt.plot(x,y_axis2,label="CNN")
    plt.legend(loc="upper left")
    plt.show()