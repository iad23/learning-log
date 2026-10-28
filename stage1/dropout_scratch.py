import os
import torch
import matplotlib.pyplot as plt
from torch import nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

# ---------- 超参数（和教材 4.6 一致，方便和简洁版对账） ----------
num_inputs, num_outputs = 784, 10
num_hiddens1, num_hiddens2 = 256, 256
dropout1, dropout2 = 0.2, 0.5      # 丢弃概率：第一层丢 20%，第二层丢 50%
batch_size, num_epochs, lr = 256, 10, 0.5

# ---------- 数据：Fashion-MNIST ----------
def load_data(batch_size):
    transform = transforms.ToTensor()
    mnist_train = datasets.FashionMNIST(root='../data', train=True,
                                        transform=transform, download=True)
    mnist_test = datasets.FashionMNIST(root='../data', train=False,
                                       transform=transform, download=True)
    # num_workers=0：Windows 上开多进程会套娃报错，3.6 踩过这个坑
    train_iter = DataLoader(mnist_train, batch_size=batch_size,
                            shuffle=True, num_workers=0)
    test_iter = DataLoader(mnist_test, batch_size=batch_size,
                           shuffle=False, num_workers=0)
    return train_iter, test_iter

# ---------- dropout 层：教材逻辑原样，注释补全 ----------
def dropout_layer(X,dropout):
    assert 0 <= dropout <=1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    # 每个元素掷一次骰子：rand 出 [0,1) 随机数，> dropout 的算"活下来"，
    # 活下来的概率正好是 1 - dropout；.float() 把 True/False 变成 1.0/0.0
    mask = (torch.rand(X.shape) > dropout).float()
    # mask * X 把丢掉的位置清零；再除以 (1 - dropout) 给幸存者发补偿，
    # 保证期望不变（单次总和是随机的，长期平均和没丢时一样）——inverted dropout
    return mask * X / (1.0 - dropout)

# ---------- 网络：手写 class 版 ----------
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2):
        super().__init__()   # 先让父类 nn.Module 装好底盘（参数账本等）
        self.num_inputs = num_inputs   # 存进自己的储物柜，forward 还要用
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()
        # 注意：这里没有 self.training = ... 这行。
        # super().__init__() 已经替你建好这个开关了，默认 True（训练模式），
        # 用 net.train() / net.eval() 切换，forward 里读的就是它

    def forward(self,X):
        H1 = self.relu(self.lin1(X.reshape((-1,self.num_inputs))))
        if self.training:
            H1 = dropout_layer(H1,dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2,dropout2)
        out = self.lin3(H2)
        return out

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)

# ---------- 训练 / 评估 ----------
def train_epoch(net, train_iter, loss_fn, updater):
    net.train()                       # 切回训练模式：打开 dropout（评估后别忘切回来）
    total_loss, total_correct, total = 0.0, 0, 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss_fn(y_hat, y)         # CrossEntropyLoss 默认 mean
        updater.zero_grad()
        l.backward()
        updater.step()
        total_loss += float(l) * len(y)   # mean 乘回 batch 大小攒总和（合同！）
        total_correct += (y_hat.argmax(dim=1) == y).sum().item()
        total += len(y)
    return total_loss / total, total_correct / total

def evaluate_accuracy(net, data_iter):
    net.eval()                        # 切到评估模式：关掉 dropout（本节第一大坑）
    total_correct, total = 0, 0
    with torch.no_grad():             # 评估不算梯度，省显存省时间
        for X, y in data_iter:
            total_correct += (net(X).argmax(dim=1) == y).sum().item()
            total += len(y)
    return total_correct / total

# ---------- 开跑 ----------
if __name__ == '__main__':
    train_iter, test_iter = load_data(batch_size)
    net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
    net.apply(init_weights)           # 把 init_weights 递给每一层，挨个执行
    loss_fn = nn.CrossEntropyLoss()
    updater = torch.optim.SGD(net.parameters(), lr=lr)

    train_losses, train_accs, test_accs = [], [], []
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, train_iter, loss_fn, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(f'epoch {epoch + 1}: loss {train_loss:.3f}, '
              f'train acc {train_acc:.3f}, test acc {test_acc:.3f}')

    # ---------- 画图存盘 ----------
    os.makedirs('picture6', exist_ok=True)
    epochs = range(1, num_epochs + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label='train loss')
    plt.plot(epochs, train_accs, label='train acc')
    plt.plot(epochs, test_accs, label='test acc')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('picture6/dropout_scratch.png')
    print('图已存到 picture6/dropout_scratch.png')

    # ---------- 验货（阈值留足余量，看量级不对劲就停下来查） ----------
    assert train_accs[-1] > 0.78, 'train acc 偏低，检查 lr 和 dropout 概率'
    assert test_accs[-1] > 0.78, 'test acc 偏低，检查 eval 是否关掉了 dropout'
    print('assert 全过')
