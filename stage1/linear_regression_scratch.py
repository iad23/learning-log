import random
import torch
from d2l import torch as d2l

# ============================================================
# 3.2.1 生成数据集：先定答案，再出题（合成数据）
# ============================================================
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声 的人工数据集"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    # torch.normal(均值, 标准差, 形状)：从正态分布里随机采样
    # (num_examples, len(w)) = (1000, 2)：1000 个样本，每个样本 2 个特征
    # shape 就是"每个维度有多大"：元组里有几个数就是几维，(1000,2) 即 1000 行 2 列
    y = torch.matmul(X, w) + b
    # torch.matmul 是矩阵乘法（行乘列再求和），X(1000,2) @ w(2,) 得到 (1000,)
    # 注意：这里的 * 不是一回事——* 是逐元素相乘，@ 才是矩阵乘法
    # + b 时 b 是标量，广播机制自动把它加到每个元素上
    y += torch.normal(0, 0.01, y.shape)
    # 撒一把噪声：均值 0、标准差 0.01，形状和 y 一样，逐元素加上
    return X, y.reshape((-1, 1))
    # y 原来是 (1000,) 一维，reshape 成 (1000, 1) 二维列向量
    # -1 的意思是"这一维你自己算"：总共 1000 个数，另一维是 1，所以是 (1000,1)

true_w = torch.tensor([2,-3.4])
true_b = 4.2
features,labels = synthetic_data(true_w,true_b,1000)
# features 形状 (1000, 2)：1000 个样本，每个 2 个特征
# labels 形状 (1000, 1)：每个样本对应 1 个真实房价
print('feature 的第0个样本： ',features[0])
print('labels 的第0个标签： ',labels[0])

# ============================================================
# 3.2.2 读取数据集：发牌员，一批批地发
# ============================================================
def data_iter(batch_size,features,labels):
    num_examples = len(features) # 样本总数 
    indices = list(range(num_examples)) # range(1000) 是懒序列，不真正存 1000 个数；list() 转成真列表才能 shuffle
    random.shuffle(indices) #打乱顺序
    for i in range(0,num_examples,batch_size): #起点，终点，步长（左闭右开）
        batch_indices = torch.tensor(indices[i:min(i + batch_size,num_examples)])
        # yield：生成器！每次吐一批，调用方拿完再回来要下一批
        # features[batch_indices] 是花式索引：按下标列表一次取多行
        yield features[batch_indices], labels[batch_indices]

batch_size = 10 # 超参数：每批 10 个样本

# 看一批长啥样（break 只看第一批；去掉 break 就遍历全部 100 批）
for X,y in data_iter(batch_size,features,labels):
    print(X,'\n',y)
    break
# X：(10, 2)，这批的输入；y：(10, 1)，这批的答案

# ============================================================
# 3.2.3 初始化模型参数：w 小随机，b 置零
# ============================================================
# w 用小随机值：torch.normal(0, 0.01, size=(2,1)) 从均值 0、标准差 0.01 的正态分布采样
#   为啥不能全零？对称性灾难——全零的话每个权重学到的东西永远一样，模型学不出差别
# b 用零：单个标量，没有对称性问题，零初始化无妨
# requires_grad=True：开启自动微分，torch 会记录计算图、算梯度
w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

# ============================================================
# 3.2.4 定义模型 linreg：算预测 y_hat（注意不是 lr！）
# ============================================================
# linreg = linear regression（线性回归）的缩写，是模型函数
# lr = learning rate（学习率）的缩写，是超参数。俩只是撞脸，毫无关系！
def linreg(X,w,b):
    """线性回归模型：y_hat = Xw + b"""
    # torch.matmul(X, w)：矩阵乘法。(10,2) @ (2,1) → (10,1)
    #   每个样本的一行和 w 做点积：y_hat[i] = X[i,0]*w[0] + X[i,1]*w[1]
    #   内维的 2 被"加掉"了，所以结果里没有 2
    #   X @ w 和 torch.matmul(X, w) 等价；* 是逐元素相乘，别混！
    # + b：广播，(10,1) + 标量，每行都加 b
    return torch.matmul(X,w) + b

# ============================================================
# 3.2.5 定义损失函数 squared_loss：扣分制
# ============================================================
# loss 衡量"预测 y_hat 和答案 y 差多远"，差越远扣分越多。训练的目标就是少扣分。
def squared_loss(y_hat,y):
    """均方损失：((y_hat - y)^2) / 2"""
    # y.reshape(y_hat.shape)：把 y 对齐到 y_hat 的形状，防广播出意外
    # ** 2：平方，差得越大扣分越多（且恒为正）
    # / 2：没深意，纯粹求导时把 2 约掉，数学上方便
    return (y_hat -  y.reshape(y_hat.shape))**2/2

# ============================================================
# 3.2.6 定义优化算法 sgd：走一步
# ============================================================
# SGD = Stochastic Gradient Descent（随机梯度下降）
#   Stochastic（随机）：每批样本随机抽；Gradient（梯度）：用梯度定方向；
#   Descent（下降）：沿梯度反方向走一小步，让 loss 下降
def sgd(params,lr,batch_size):
    """小批量随机梯度下降：更新参数"""
    # with torch.no_grad()：关掉梯度记录。调参不是模型计算，不用记；
    #   记了的话计算图每轮越滚越大。with 块内有效，出了块恢复。
    with torch.no_grad():
        # params 是 [w, b]，逐个更新。param 和 w 是贴在同一个对象上的两个标签
        for param in params: # params = [w, b]，逐个更新
            param -= lr*param.grad / batch_size
            # param.grad 是 backward() 存好的梯度（w.grad 形状 (2,1)，b.grad 形状 (1,)）
            # / batch_size：把"和"的梯度变成平均梯度，步长不随 batch 大小漂移
            # -= 是就地操作，直接改对象里的数据，外面的 w、b 同步变，不用 return
            param.grad.zero_()   # 清空梯度：PyTorch 默认累积，不清会越叠越大

# ============================================================
# 3.2.7 训练：双层循环，内层学、外层轮末考
# ============================================================
lr = 3  # 超参数：学习率
num_epochs = 0.03  # 超参数：数据集刷 3 遍
net = linreg   # 函数别名：net 指向 linreg。换模型时只改指向，循环不动
loss = squared_loss

for epoch in range(num_epochs): # 外层：刷 3 轮。每轮重新 shuffle 发牌
    for X,y in data_iter(batch_size,features,labels):
        l = loss(net(X,w,b),y)
        # 反向：l.sum() 聚成标量（backward 只能从标量出发）
        # backward() 把梯度存进 w.grad / b.grad
        l.sum().backward()
        # 更新：用梯度走一步，再清零。注意顺序不能反：先算梯度，再更新
        sgd([w,b],lr,batch_size)
    # 每轮结束考一次：with 和内层 for 同级，每 epoch 跑一次
    # 用全量 1000 样本算 loss，看看这轮学得咋样（只看不练，所以 no_grad）
    with torch.no_grad():
        train_l = loss(net(features,w,b),labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# 对答案：估计误差 = 真实值 - 学到的值
print(f'w 的估计误差：{true_w - w.reshape(true_w.shape)}')
print(f'b 的估计误差：{true_b - b}')
# 误差约 1e-4 量级：模型几乎完美还原了上帝参数