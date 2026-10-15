import torch
from torch import nn  # nn = torch.nn，神经网络组件的大本营
from torch.utils import data  # DataLoader（发牌员）住在这里
from d2l import torch as d2l

# ============================================================
# 3.3.1 生成数据集：和 3.2 一模一样（上帝先定答案，再出题）
# ============================================================
true_w = torch.tensor([2,-3.4]) #两个权重w
true_b = 4.2 #偏置，标量，b

# synthetic_data 干的事和 3.2 完全相同：X ~ N(0,1) 采样 1000 个样本，
# y = X @ true_w + true_b 再加一点噪声，最后把 y 整成 (1000,1)
features,labels = d2l.synthetic_data(true_w,true_b,1000)
# features：(1000, 2)，输入 x；  labels：(1000, 1)，标准答案 y

# ============================================================
# 3.3.2 读取数据：手写 data_iter 换成现成的 DataLoader
# ============================================================
def load_array(data_arrays,batch_size,is_train=True):
    """构造一个 PyTorch 数据迭代器（书 3.3 现场定义的函数）"""
    # TensorDataset：把 (features, labels) 打包成"一个个样本"的数据集
    #   *data_arrays 是解包：(features, labels) → 两个参数传进去
    dataset = data.TensorDataset(*data_arrays)
    # DataLoader：现成的发牌员。batch_size=每批几张牌，shuffle=True 每轮洗牌
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

batch_size = 10   # 超参数：每批 10 个样本
data_iter = load_array((features, labels), batch_size)
# 用法和 3.2 的 data_iter 一样：for X, y in data_iter，每轮自动 shuffle

# ============================================================
# 3.3.3 定义模型：w、b 从外部变量"住进" net 对象里
# ============================================================
# nn.Linear(2, 1)：现成的线性层，2 个输入 → 1 个输出，干的就是 Xw + b 这件事。
#   w、b 不用你建，住在它肚子里：
#     net[0].weight：形状 (1,2)——注意是转置存的！
#       PyTorch 约定 weight 是 (out_features, in_features)，计算时内部自动转置
#     net[0].bias：形状 (1,)
# nn.Sequential(...)：串层的容器，把层按顺序装进去，X 从头穿到尾。
#   现在里面只有一层，以后多层网络就往里接着加。
net = nn.Sequential(nn.Linear(2,1))
# net(X) 把对象当函数调：nn.Module 定义了 __call__，实际执行的是 forward(X)，
# 这是 Python"可调用对象"机制。对比 3.2：linreg(X, w, b) 要手传 w、b，
# 3.3 里 w、b 住进 net 了，net(X) 一个参数就够——这就是"调包"的爽点。

# 初始化：和 3.2 一模一样的值——w 小随机打破对称性，b 置零
#   net[0]：取出容器里第 0 层（就是那个 Linear）
#   .weight.data：拿到参数张量本身。.data 是"去掉 autograd 包装"的裸数据，
#     初始化是直接写值、不是模型计算，所以写裸数据，不用记梯度
#   normal_ / fill_：下划线后缀 = in-place，就地改（和 3.2 里 zero_() 一个路数）
net[0].weight.data.normal_(0, 0.01)  # w ~ N(0, 0.01²)，小随机
net[0].bias.data.fill_(0)            # b = 0，单个标量无对称性问题
# ============================================================
# 3.3.4 定义损失函数：手写 squared_loss 换成 nn.MSELoss
# ============================================================
# ⚠️ 坑 1：MSELoss 默认 reduction='mean'（取平均），但和 3.2 等价，lr 照抄 0.03 就行！
#   3.2：l.sum().backward() → 梯度 = Σ∇lᵢ；手写 sgd 里 -= lr·grad / batch_size
#        → 实际步长 = lr · mean(∇lᵢ)
#   3.3：MSELoss 先把 10 个样本的损失取平均 → l.backward() 直接给出 mean(∇lᵢ)；
#        优化器做 -= lr·grad → 步长 = lr · mean(∇lᵢ)
#   两边步长一致，所以 3.3 照抄 lr=0.03，不是巧合。
#   （较真版：3.2 手写版损失里还有个 /2（求导时约掉那个 2），MSELoss 没有，
#    所以 3.3 的梯度其实是 2 倍——相当于有效 lr 翻倍。但你早上刚实测过，
#    lr=0.3 在这题上都不发散，2 倍完全在安全区内，书里就这么写的，不用管。）
loss = nn.MSELoss()
# 注意：3.3 的 l 已经是标量（一个数），后面直接 l.backward()，不用 .sum() 了——
# 这正是 3.2 对照表里"期末考试"那行说的：l 是标量，打印时也不用 .mean()。


# ============================================================
# 3.3.5 定义优化算法：手写 sgd 换成 torch.optim.SGD
# ============================================================
# net.parameters()：把 net 肚子里所有要学的东西（w、b）交出来，打包给优化器
# lr=0.03：和 3.2 同一个意思，照抄（见坑 1）
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
# 循环里固定三句，顺序不能乱：
#   ① trainer.zero_grad()：清零梯度（⚠️ 坑 2，见训练循环）
#   ② l.backward()：反向传播，梯度存进 net 里每个参数的 .grad
#   ③ trainer.step()：走一步，做 -= lr·grad（就是 3.2 手写 sgd 干的事）


# ============================================================
# 3.3.6 训练：双层循环，和 3.2 长得一样
# ============================================================
num_epochs = 3  # 超参数：数据集刷 3 遍（人定的）
for epoch in range(num_epochs):  # 外层：刷 3 轮，每轮 DataLoader 自动 shuffle
    for X,y in data_iter:  # 内层：100 批，走 100 步
        # ① 前向：算这批的预测和损失。l 是标量，直接 backward，不用 .sum()
        l = loss(net(X), y)
        # ② ⚠️ 坑 2：先清零，再 backward，顺序不能反！
        #    3.2 里清零是手写 sgd 内部顺手做的（param.grad.zero_()）；
        #    3.3 里必须自己在 backward() 之前调 trainer.zero_grad()，
        #    否则梯度累积——每批的梯度叠在上一批上，越学越歪。
        #    这正是 3.2 里讲过的"梯度默认累积"问题，调包版也不会替你收拾。
        trainer.zero_grad() #清零
        l.backward() #反向求梯度
        trainer.step() #优化一步
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

# 交卷对答案：和 3.2 一样，上帝答案在手，直接验算学得准不准
#   net[0].weight.data：形状 (1,2)，reshape 成 (2,) 再和 true_w 比
w = net[0].weight.data
print(f'w的估计误差：{true_w - w.reshape(true_w.shape)}')
b = net[0].bias.data
print(f'b的估计误差：{true_b - b}')
# 误差应该和 3.2 一样是 1e-4 量级——调包版学出来的东西，和手搓版一样准。
# 这就是 3.3 想告诉你的：循环还是那个循环，换的只是零件。
