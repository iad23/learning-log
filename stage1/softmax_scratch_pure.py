import torch
import torchvision
from torchvision import transforms
from torch.utils import data
import matplotlib.pyplot as plt
import os
os.makedirs('picture', exist_ok=True)   # 没有就建，有了就跳过
# ============================================================
# 3.5 回顾：读数据（d2l.load_data_fashion_mnist 的原生版）
# ============================================================
def get_dataloader_workers():
    # 4 个"帮厨"进程：后台读数据、做 ToTensor、组 batch，不让训练干等。
    # Windows 下如果运行时卡住，把这里改成 0 再跑（单进程读数据，最稳）。
    return 0
def load_data_fashion_mnist(batch_size,resize=None):
    """下载 Fashion-MNIST → 预处理 → 包成两个 DataLoader"""
    trans = [transforms.ToTensor()]   # PIL 图 → tensor，同时 0~255 归一化到 0~1
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans = transforms.Compose(trans)   # 把变换串成一条流水线

    mnist_train = torchvision.datasets.FashionMNIST(
        root='../data',train=True,transform=trans,download=True)  # 6 万训练
    mnist_test = torchvision.datasets.FashionMNIST(
        root='../data',train=False,transform=trans,download=True) # 1 万测试
    # 训练集 shuffle=True（打乱，防背顺序），测试集 shuffle=False（顺序固定好复现）
    return (data.DataLoader(mnist_train,batch_size,shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test,batch_size,shuffle=False,
                            num_workers=get_dataloader_workers()))


batch_size = 256   # 每批 256 张图
train_iter,test_iter = load_data_fashion_mnist(batch_size)
# 每批吐 (X, y)：X 形状 (256,1,28,28)——1 是灰度通道数；y 是 256 个整数标签（0~9）

# ============================================================
# 3.6.1 初始化模型参数
# ============================================================
num_imputs = 784  # 28*28：每张图展平成 784 个数
num_outputs = 10   # 10 个类别：输出 10 个"这是第 k 类"的打分
W = torch.normal(0,0.01,size=(num_imputs,num_outputs),requires_grad = True)
b = torch.zeros(num_outputs,requires_grad = True)
# W：(784,10)——784 个输入特征 × 10 个类别，每个类别有自己的一套权重。
#   torch.normal(均值, 标准差, size)：第一个 0 是均值，第二个 0.01 是标准差
#   （10-19 刚踩过的坑：有两个数字，别把那个 0 当没写）。
# b：(10,)——每个类别一个偏置，广播加到每一行上。
# requires_grad=True：标记"这是学生，backward 时给它算梯度"。

# ============================================================
# 3.6.2 定义 softmax 操作：打分 → 概率
# ============================================================
def softmax(X):
    """X：(256,10) 的打分表 → (256,10) 的概率表，每行和为 1"""
    X_exp = torch.exp(X)
    partition = X_exp.sum(1,keepdim=True)
    return X_exp / partition   # 广播相除：每行变成概率分布
# keepdim=True：求和后留个 1 占位，(256,1) 才能和 (256,10) 广播相除。
# 直觉：打分最高的类别，exp 后占比最大——softmax 就是"把打分翻译成有多确定"。

# ============================================================
# 3.6.3 定义模型
# ============================================================
def net(X):
    # X：(256,1,28,28) → reshape 成 (256,784)：拍扁，-1 表示"这维自动算"
    # X @ W：(256,784)@(784,10) → (256,10)，10 个类别的打分（matmul=@，别和 * 搞混）
    # + b：(10,) 广播到每一行；softmax：打分 → 概率
    return softmax(torch.matmul(X.reshape((-1,W.shape[0])),W)+b)

# ============================================================
# 3.6.4 定义损失函数：交叉熵（惊讶函数）
# ============================================================
def cross_entropy(y_hat,y):
    """y_hat：(256,10) 概率表；y：256 个整数标签。返回 256 个"扣分"。"""
    # y_hat[range(len(y_hat)), y]：花式索引——两个列表按位置配对成 (行,列) 坐标。
    #   range(256) 是行号（第 0~255 个样本，len(y_hat) 自适应 batch 大小），
    #   y 是每行的真实类别（列号）。取出 = 每个样本分给自己正确答案的概率。
    # -torch.log(...)："惊讶函数"——给正确答案 0.99 扣 0.01 分，给 0.01 扣 4.6 分。
    return -torch.log(y_hat[range(len(y_hat)),y])

# ============================================================
# 3.6.5 分类精度 + 记分牌
# ============================================================
def accuracy(y_hat,y):
    """这批猜对了几个，返回猜对的个数（注意：是个数，不是比例）"""
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y   # 逐个比"猜的"和"答案"，True=猜对
    return float(cmp.type(y.dtype).sum())

class Accumulator:
    """d2l 的小累加器原样搬来：n 个计数器，add 往里加，[i] 读数"""
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net,data_iter):
    """在整个测试集上算准确率：猜对 / 总数"""
    metric = Accumulator(2)  # 记两个数：猜对数，样本总数
    with torch.no_grad():
        for X,y in data_iter:
            metric.add(accuracy(net(X),y),y.numel())  # y.numel() = 这批几个样本
    return metric[0] / metric[1]

# ============================================================
# 3.2 回顾：手写小批量 SGD（d2l.sgd 的原生版）
# ============================================================
def sgd(params,lr,batch_size):
    """小批量随机梯度下降：param -= lr * grad / batch_size，再清零"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# ============================================================
# 3.6.6 训练
# ============================================================
def train_epoch_ch3(net,train_iter,loss,updater):
    """训一个 epoch，返回 (平均损失, 训练准确率)"""
    metric = Accumulator(3)
    for X,y in train_iter:
        y_hat = net(X)      # ① 前向：(256,10) 概率表
        l = loss(y_hat, y)  # ② 扣分：256 个 -log，一个样本一分
        l.sum().backward()  # ③ 反向：backward 只能从标量出发，先求和再反向
        updater(X.shape[0])  # ④ 更新：X.shape[0]=256 传进去，sgd 内部除 batch
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # metric[0]/metric[2] = 总损失/总样本 = 平均损失；metric[1]/metric[2] = 准确率
    return metric[0] / metric[2], metric[1] / metric[2]

class Animator:
    """d2l.Animator 的 20 行极简替代：训练"仪表盘"，每轮添三个点"""
    def __init__(self, xlabel=None, xlim=None, ylim=None, legend=None,
                 fmts=('-', 'm--', 'g-.')):
        # fmts：三条线的样式——实线、品红虚线、绿色点划线，对应 legend 三项
        self.fig, self.ax = plt.subplots(figsize=(3.5, 2.5))
        self.ax.set_xlabel(xlabel)
        if xlim:
            self.ax.set_xlim(xlim)
        if ylim:
            self.ax.set_ylim(ylim)
        self.legend, self.fmts = legend, fmts
        self.X, self.Y = [], []  # 存历史点：三条线各一个列表
        plt.ion()   # 交互模式：show() 不阻塞，训练继续跑，图实时刷新
        plt.show()

    def add(self, x, y):
        # x：epoch（从 1 开始）；y：(train_loss, train_acc, test_acc) 三元组
        if not self.X:  # 第一轮：建三个空列表
            self.X = [[] for _ in y]
            self.Y = [[] for _ in y]
        for i, b in enumerate(y):
            self.X[i].append(x)
            self.Y[i].append(b)
        self.ax.cla()  # 清空重画（和 d2l.Animator 一个套路）
        for xs, ys, fmt in zip(self.X, self.Y, self.fmts):
            self.ax.plot(xs, ys, fmt)
        self.ax.legend(self.legend)
        plt.pause(0.01)  # 给 GUI 一点时间把新图刷出来

def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
    """训 num_epochs 轮：每轮"训练→考试→画图"，最后交卷质检"""
    animator = Animator(xlabel='epoch',xlim=[1,num_epochs],ylim=[0.3,0.9],
                        legend=['train loss','train acc','test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net,train_iter,loss,updater)
        test_acc = evaluate_accuracy(net,test_iter)   # 每轮结束拿测试集"考一次"
        # train_metrics=(loss, acc) 是元组，+(test_acc,) 拼成三元组，对上三条线。
        # 注意 (test_acc,) 那个逗号：没它就不是元组，拼不起来。
        animator.add(epoch+1,train_metrics + (test_acc,))
    # 交卷标准：loss < 0.5，训练/测试准确率 > 0.7。断言炸了 = 前面哪个零件写错了。
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5,train_loss
    assert train_acc <=1 and train_acc > 0.7,train_acc
    assert test_acc <= 1 and test_acc>0.7,test_acc

# ============================================================
# 定义优化器（手写版）+ 开训
# ============================================================
lr = 0.1  # 学习率 0.1
def updater(batch_size):
     # 闭包：把 W、b、lr 打包带进去，train_epoch_ch3 里调 updater(256) 就行。
     return sgd([W,b],lr,batch_size)

num_epochs = 10
train_ch3(net,train_iter,test_iter,cross_entropy,num_epochs,updater)
# 跑完应该看到：train loss 一路下到 ~0.3，train acc 和 test acc 爬到 0.8+。
# test acc 比 train acc 低一点正常——考试题没见过嘛。
plt.savefig('picture/training_curve.png', dpi=150)
plt.ioff()   # 关交互模式
plt.show()   # 训练结束：把仪表盘定住，不关窗口程序不退出

# ============================================================
# 3.6.7 预测：抓几张出来看看猜得对不对（d2l.show_images 的原生版）
# ============================================================
def get_fashion_mnist_labels(labels):
    """0~9 → 英文名：d2l.get_fashion_mnist_labels 的替代，就是个查表"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """d2l.show_images 的替代：把 n 张灰度图排成 num_rows×num_cols 网格"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = [axes] if num_rows * num_cols == 1 else axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        # cmap='gray'：灰度显示。默认 colormap 会把单通道图染成紫黄色（3.5 踩过的坑）
        ax.imshow(img.numpy(), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])  # 藏刻度，图干净
        if titles:
            ax.set_title(titles[i], fontsize=9)
    return axes

def predict_ch3(net, test_iter, n=6):
    """从测试集里抓 n 张，画出来看看模型猜得对不对"""
    for X, y in test_iter:
        break  # 只拿第一批：DataLoader 是迭代器不支持 [0]，for+break 是惯用写法
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))  # 模型觉得最像的类
    # 标题两行：真答案 \n 模型猜的。猜错的一眼揪出来。
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    # X[0:n].reshape((n,28,28))：把通道维 1 挤掉，imshow 要 (H,W)
    show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    plt.savefig('picture/predictions.png', dpi=150)
    plt.show()  # 定住看图


predict_ch3(net, test_iter)
