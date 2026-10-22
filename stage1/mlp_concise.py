import os
import torch
import matplotlib.pyplot as plt
from torch import nn
from mytools import *

os.makedirs('picture3',exist_ok=True)

batch_size = 256
train_iter,test_iter = load_data_fashion_mnist(batch_size)

num_inputs,num_outputs,num_hiddens = 784,10,256

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784,256),
                    nn.ReLU(),
                    nn.Linear(256,10))

def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weight)

num_epochs,lr = 10,0.1

loss = nn.CrossEntropyLoss()

updater = torch.optim.SGD(net.parameters(),lr = lr)

train_ch3(net,train_iter,test_iter,loss,num_epochs,updater)

plt.savefig('picture3/training_curve_mlp.png', dpi=150)

predict_ch3(net, test_iter)
plt.savefig('picture3/predictions_mlp.png', dpi=150)

plt.ioff()
plt.show()