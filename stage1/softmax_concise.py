import torch
import torchvision
import os
from torch import nn
import matplotlib.pyplot as plt
from torch.utils import data
from torchvision import transforms

os.makedirs('picture1',exist_ok = True)

def get_dataloader_workers():
    return 0;

def load_data_fashion_mnist(batch_size,resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(
        root='../data',train=True,transform=trans,download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root='../data',train=False,transform=trans,download=True)
    return(data.DataLoader(mnist_train,batch_size,shuffle=True,
                           num_workers=get_dataloader_workers()),
           data.DataLoader(mnist_test,batch_size,shuffle=False,
                           num_workers=get_dataloader_workers()))

batch_size=256
train_iter,test_iter = load_data_fashion_mnist(batch_size)

net = nn.Sequential(
    nn.Flatten(),   # (256,1,28,28) → (256,784)：干的是纯净版 net 里的 reshape 那步
    nn.Linear(784,10),  # 内部 = X@W.T + b：干的是纯净版的 matmul + b 那步
    # ⚠️ 这里没有 softmax 层！Linear 直接吐 logits（原始打分）。
    #    softmax 已经搬进下面的 CrossEntropyLoss 里了；
    #    你要是再套一层 softmax，就是"双重 softmax"，直接学废（对照表标红那条）。11
)

# ============================================================
# 3.7.2 初始化：apply 巡检，只重填 Linear 的 weight
# ============================================================
def init_weights(m):
    if type(m) == nn.Linear:
        # = 纯净版的 torch.normal(0, 0.01)，只不过这次是原地重填这张表（结尾 _ 的含义）
        nn.init.normal_(m.weight,std=0.01)

net.apply(init_weights)

# ============================================================
# 3.7.3 损失：官方交叉熵，走"默认 mean"套餐
# ============================================================
loss = nn.CrossEntropyLoss()
# 它吃的是 logits（net 的原始打分），不是概率——内部一步做完：
#   log_softmax → 取正确类别那一项 → 对全批求平均。
# 默认 reduction='mean'：返回 1 个标量（256 个样本损失的平均）。
# 【合同】（你刚问通的那套）：mean 已经除过 batch 了，所以配套是 torch.optim：
#   backward 时 l 本身是标量，直接 l.backward()，不用 sum，也没人再除 batch。
# 要是改成 reduction='none'，它就退化成纯净版手写 cross_entropy（返回 256 个数），
#   那必须回去配 sum().backward() + 手写 sgd。两套餐选一套到底，别混搭。

# ============================================================
# 3.7.4 优化器：训练前"雇好人"，花名册和步长一次交代清
# ============================================================
trainer = torch.optim.SGD(net.parameters(),lr=0.1)

# ============================================================
# 记分牌 + 精度 + 评估：原样搬，只有一处要注意（见 accuracy 注释）
# ============================================================
def accuracy(y_hat,y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp=y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

class Accumulator:
    """n 个计数器：add 往里加，[i] 读数（和纯净版一模一样）"""
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def train_epoch_ch3(net,train_iter,loss,trainer):
    metric = Accumulator(3)
    for X,y in train_iter:
        y_hat = net(X)
        l = loss(y_hat,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
        metric.add(float(l) * len(y), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1]/metric[2]

class Animator:
    """训练"仪表盘"：每轮添三个点（和纯净版一模一样）"""
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

num_epochs = 10
def train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, trainer)
        test_acc = evaluate_accuracy(net, test_iter)   # 每轮结束拿测试集"考一次"
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
# 跑完应该看到：train loss 一路下到 ~0.45，train/test acc 爬到 0.84~0.85，
# 和 3.6 纯净版基本同分——换零件不换成绩，变的只是写法。
plt.savefig('picture1/training_curve_concise.png', dpi=150)   # 先拍照
plt.ioff()   # 关交互模式
plt.show()   # 再堵门：把仪表盘定住，关窗才往下走

# ============================================================
# 3.7.6 预测：原样搬（net 现在吐 logits，argmax 照旧，先变概率纯属多余）
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
    """从测试集里抓 n 张，画出来看看模型猜得对不对"""
    for X, y in test_iter:
        break  # 只拿第一批（DataLoader 是迭代器，for+break 是惯用写法）
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))  # 对 logits 直接 argmax
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    plt.savefig('picture1/predictions_concise.png', dpi=150)   # 先拍照
    plt.show()  # 再堵门看图

predict_ch3(net, test_iter)
