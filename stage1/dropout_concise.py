import os
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ---------- 超参数（与 scratch 版完全一致） ----------
num_inputs, num_outputs = 784, 10
num_hiddens1, num_hiddens2 = 256, 256
dropout1, dropout2 = 0.2, 0.5      # nn.Dropout 的参数同样是"丢弃概率"，别和保留概率搞反
batch_size, num_epochs, lr = 256, 10, 0.5

# ---------- 数据：Fashion-MNIST（与 scratch 版相同） ----------
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

# ---------- 网络：Sequential 乐高版 ----------
net = nn.Sequential(
    nn.Flatten(),                        # 代替手写的 reshape((-1, 784))：把图片压平
    nn.Linear(num_inputs, num_hiddens1),
    nn.ReLU(),
    nn.Dropout(dropout1),                # 在第一个全连接层之后添加一个 dropout 层
    nn.Linear(num_hiddens1, num_hiddens2),
    nn.ReLU(),
    nn.Dropout(dropout2),                # 在第二个全连接层之后添加一个 dropout 层
    nn.Linear(num_hiddens2, num_outputs) # 输出层不加 relu、不加 dropout
)

# ---------- 权重初始化：和 scratch 版同一个函数、同一行 apply ----------
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)   # 原地重抽成 N(0, 0.01)，对齐起点

net.apply(init_weights)                  # 把 init_weights 递给每一层，挨个执行

# ---------- 训练 / 评估（与 scratch 版相同） ----------
def train_epoch(net, train_iter, loss_fn, updater):
    net.train()                       # 切回训练模式：nn.Dropout 开始随机丢
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
    net.eval()                        # 切到评估模式：nn.Dropout 自动全关（忘了它，测试结果每次都不一样）
    total_correct, total = 0, 0
    with torch.no_grad():             # 评估不算梯度，省显存省时间
        for X, y in data_iter:
            total_correct += (net(X).argmax(dim=1) == y).sum().item()
            total += len(y)
    return total_correct / total

# ---------- 开跑 ----------
if __name__ == '__main__':
    train_iter, test_iter = load_data(batch_size)
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
    plt.savefig('picture6/dropout_concise.png')
    print('图已存到 picture6/dropout_concise.png')

    # ---------- 验货（阈值留足余量，看量级不对劲就停下来查） ----------
    assert train_accs[-1] > 0.78, 'train acc 偏低，检查 lr 和 dropout 概率'
    assert test_accs[-1] > 0.78, 'test acc 偏低，多半是忘了切 eval'
    print('assert 全过')
