"""
mytools.py —— 我自己的迷你 d2l
从 3.7 跑通的 softmax_concise.py 里拆出来的脚手架：数据加载、记分牌、
仪表盘、训练/评估/预测——全是与"模型长啥样"无关的通用件，逻辑原样未改。

用法：本文件和实验文件放同一目录（stage1），实验文件开头写：
    from mytools import load_data_fashion_mnist, train_ch3, predict_ch3

【合同】本文件的 train_ch3 按"默认 mean"套餐写：loss 必须用
nn.CrossEntropyLoss()（默认 reduction='mean'），优化器用 torch.optim
（如 torch.optim.SGD(params, lr=...) 或 SGD(net.parameters(), lr=...)）。
不要把 loss 改成 reduction='none'——那是另一套合同，混搭会让损失记账
被缩小约 batch 倍、assert 假通过（3.7 踩过的坑）。
"""

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils import data
from torchvision import transforms


# ============================================================
# 数据加载（3.5 那一套）
# ============================================================
def get_dataloader_workers():
    # 跑通时的结论：Windows 下用 0（主进程自己读图），不开"帮厨"进程最稳。
    return 0

def load_data_fashion_mnist(batch_size, resize=None):
    """下载 Fashion-MNIST → 预处理 → 包成两个 DataLoader"""
    trans = [transforms.ToTensor()]   # PIL 图 → tensor，同时 0~255 归一化到 0~1
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)   # 把变换串成一条流水线

    mnist_train = torchvision.datasets.FashionMNIST(
        root='../data', train=True, transform=trans, download=True)   # 6 万训练
    mnist_test = torchvision.datasets.FashionMNIST(
        root='../data', train=False, transform=trans, download=True)  # 1 万测试
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


# ============================================================
# 记分牌 + 评估
# ============================================================
def accuracy(y_hat, y):
    """这批猜对了几个（个数，不是比例）。y_hat 是 logits 也没关系：
    softmax 单调，对 logits 取 argmax 和对概率取 argmax 结果一致。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

class Accumulator:
    """n 个计数器：add 往里加，[i] 读数"""
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net, data_iter):
    """在整个测试集上算准确率：猜对 / 总数"""
    metric = Accumulator(2)  # 猜对数，样本总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


# ============================================================
# 训练（官方套餐：loss 默认 mean + torch.optim 优化器）
# ============================================================
def train_epoch_ch3(net, train_iter, loss, trainer):
    """训一个 epoch，返回 (平均损失, 训练准确率)"""
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)       # 1. 前向：(batch,10) 的 logits
        l = loss(y_hat, y)   # 2. 损失：标量（mean 已在 loss 内部算好）
        trainer.zero_grad()  # 3. 清零：把上一批的梯度账本抹掉
        l.backward()         # 4. 反向：l 是标量，直接 backward，不用再 sum
        trainer.step()       # 5. 更新：照各参数的 .grad 挪参数
        # 记账：l 是"平均损失"，乘回本批样本数才是"总损失"，
        # metric[0] 攒总损失、最后除以总样本数才是真正的平均损失。
        metric.add(float(l) * len(y), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

class Animator:
    """训练"仪表盘"：每轮添三个点"""
    def __init__(self, xlabel=None, xlim=None, ylim=None, legend=None,
                 fmts=('-', 'm--', 'g-.')):
        self.fig, self.ax = plt.subplots(figsize=(3.5, 2.5))
        self.ax.set_xlabel(xlabel)
        if xlim:
            self.ax.set_xlim(xlim)
        if ylim:
            self.ax.set_ylim(ylim)
        self.legend, self.fmts = legend, fmts
        self.X, self.Y = [], []  # 存历史点：三条线各一个列表
        plt.ion()   # 交互模式：show() 不阻塞，训练继续跑
        plt.show()

    def add(self, x, y):
        # x：epoch；y：(train_loss, train_acc, test_acc) 三元组
        if not self.X:  # 第一轮：建三个空列表
            self.X = [[] for _ in y]
            self.Y = [[] for _ in y]
        for i, b in enumerate(y):
            self.X[i].append(x)
            self.Y[i].append(b)
        self.ax.cla()  # 清空重画
        for xs, ys, fmt in zip(self.X, self.Y, self.fmts):
            self.ax.plot(xs, ys, fmt)
        self.ax.legend(self.legend)
        plt.pause(0.01)  # 给 GUI 一点时间把新图刷出来

def train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer):
    """训 num_epochs 轮：每轮"训练→考试→画图"，最后交卷质检"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, trainer)
        test_acc = evaluate_accuracy(net, test_iter)   # 每轮结束拿测试集"考一次"
        animator.add(epoch + 1, train_metrics + (test_acc,))
    # 交卷标准：loss < 0.5，训练/测试准确率 > 0.7
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


# ============================================================
# 预测展示
# ============================================================
def get_fashion_mnist_labels(labels):
    """0~9 → 英文名，就是个查表"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """把 n 张灰度图排成 num_rows×num_cols 网格"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = [axes] if num_rows * num_cols == 1 else axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.numpy(), cmap='gray')   # cmap='gray'：灰度显示，别染成紫黄
        ax.set_xticks([])
        ax.set_yticks([])  # 藏刻度，图干净
        if titles:
            ax.set_title(titles[i], fontsize=9)
    return axes

def predict_ch3(net, test_iter, n=6):
    """从测试集里抓 n 张，画出来看看模型猜得对不对。
    注意：只画图、不存图；要存图在实验文件里自己 plt.savefig。"""
    for X, y in test_iter:
        break  # 只拿第一批（DataLoader 是迭代器，for+break 是惯用写法）
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))  # 对 logits 直接 argmax
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    plt.show()
