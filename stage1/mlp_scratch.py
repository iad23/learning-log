import torch
import torchvision
import os
import matplotlib.pyplot as plt
from torch import nn
from mytools import *

os.makedirs('picture2', exist_ok=True)

batch_size = 256   # 每批 256 张图（注意：这个 256 和下面的隐藏单元 256 纯属撞数）
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs,num_outputs,num_hiddens = 784,10,256

W1 = nn.Parameter(torch.randn(
    num_inputs,num_hiddens,requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens,requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens,num_outputs,requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs,requires_grad=True))

params = [W1,b1,W2,b2] # 花名册：等下交给优化器统一管理，就是 3.7 里 net.parameters() 的手写版

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(a,X)

def net(X):
    X = X.reshape((-1,num_inputs)) # 拍扁：(256,1,28,28) → (256,784)（= nn.Flatten 干的活）
    H = relu(X@W1 + b1)  # 第一站：784 个像素 → 256 个隐藏特征，过 relu 折一下
    return H@W2 + b2  # 第二站：256 个特征 → 10 个分数（logits）

loss = nn.CrossEntropyLoss()

num_epochs,lr = 10,0.1
updater = torch.optim.SGD(params,lr=lr)
train_ch3(net,train_iter,test_iter,loss,num_epochs,updater)

plt.savefig('picture2/training_curve_mlp.png', dpi=150)

predict_ch3(net, test_iter)
plt.savefig('picture2/predictions_mlp.png', dpi=150)


plt.ioff()
plt.show()