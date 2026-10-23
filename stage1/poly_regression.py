import math
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils import data

max_degree = 20   # 特征表总共造 20 列（x 的 0~19 次方）
n_train,n_test = 100,100  #100个训练，100个测试

true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5,1.2,-3.4,5.6])

features = np.random.normal(size=(n_train + n_test,1))
np.random.shuffle(features)

# 把每个 x 扩成一行幂表：(200,1) 的 x 广播配上 (1,20) 的指数 arange(20)
# arange 只供应“幂次”，幂是 np.power 算出来的
poly_features = np.power(features,np.arange(max_degree).reshape(1,-1))
for i in range(max_degree):
    poly_features[:,i] /= math.gamma(i+1)

labels = np.dot(poly_features,true_w)   # 用真理算标签：(200,20) @ (20,) -> (200,)，w4 以后全 0 所以实际只有前四项出声
labels += np.random.normal(scale=0.1,size=labels.shape)

features,poly_features,labels = [ torch.tensor(x, dtype=torch.float32) for x in (features, poly_features, labels)]

def load_array(data_arrays, batch_size, is_train=True):
    """把 (特征, 标签) 打包成批次迭代器 —— 这就是 d2l.load_array 的本体，没几行。"""
    dataset = data.TensorDataset(*data_arrays)
    # num_workers=0：Windows 上开多进程会 spawn 套娃（3.6 踩过的坑），数据小没必要
    return data.DataLoader(dataset, batch_size, shuffle=is_train, num_workers=0)

def evaluate_loss(net, data_iter, loss):
    """在一个完整数据集上算平均损失。测试集上算出的就是“泛化损失”的替身。"""
    total, count = 0.0, 0
    with torch.no_grad():  # 评估不算梯度，省事又防手滑改到参数
        for X, y in data_iter:
            l = loss(net(X), y)         # mean 合同：这是这一批的人均损失
            total += float(l) * len(y)  # 乘回人数，还原成这一批的总分再入账
            count += len(y)
    return total / count

def train_epoch(net, train_iter, loss, trainer):
    total,count = 0.0,0
    for X,y in train_iter:
        l = loss(net(X),y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
        total += float(l) * len(y)
        count += len(y)
    return total / count

def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = nn.MSELoss()
    input_shape = train_features.shape[-1]    # shape[-1] = 最后一维 = 特征列数
    # bias=False：偏置的活已被特征第 0 列（x^0=1）乘 w0 干了，Linear 不再自带一个
    net = nn.Sequential(nn.Linear(input_shape,1,bias=False))
    batch_size = min(10,train_labels.shape[0])
     # 标签 reshape 成 (N, 1)，和 net 的输出形状对齐
    train_iter = load_array((train_features, train_labels.reshape(-1, 1)),
                            batch_size)
    test_iter = load_array((test_features, test_labels.reshape(-1, 1)),
                           batch_size, is_train=False)  # 测试集不打乱    
    trainer = torch.optim.SGD(net.parameters(),lr=0.01)
    epochs_seen, train_losses, test_losses = [], [], []
    for epoch in range(num_epochs):
        train_epoch(net, train_iter, loss, trainer)  # 训练每轮都做，400 轮一轮不少
        if epoch == 0 or (epoch + 1) % 20 == 0:  # 只在第 1 轮和每逢 20 轮时记一笔
            epochs_seen.append(epoch + 1)
            train_losses.append(evaluate_loss(net, train_iter, loss))
            test_losses.append(evaluate_loss(net, test_iter, loss))
    print('weight:', net[0].weight.data.numpy())
    # 画图（替代 d2l.Animator）：纵轴用对数刻度，loss 跨好几个数量级，普通刻度看不清
    plt.figure(figsize=(3.5, 2.5))
    plt.plot(epochs_seen, train_losses, label='train')
    plt.plot(epochs_seen, test_losses, label='test')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.xlim(1, num_epochs)
    plt.ylim(1e-3, 1e2)
    plt.legend()

if __name__ == '__main__':
    os.makedirs('picture4', exist_ok=True)  # 本节的图存 picture4（接着 4.2 的 picture2、4.3 的 picture3）

    # 实验一：前 2 列 = 1 阶 -> 预期欠拟合：train/test 都高
    print('--- 1 阶（2 列特征）---')
    train(poly_features[:n_train, :2], poly_features[n_train:, :2],
          labels[:n_train], labels[n_train:])
    plt.savefig('picture4/degree1.png', dpi=150)
    plt.close('all')

    # 实验二：前 4 列 = 3 阶，和真理同阶 -> 预期恰当：weight 应接近 [5, 1.2, -3.4, 5.6]
    print('--- 3 阶（4 列特征）---')
    train(poly_features[:n_train, :4], poly_features[n_train:, :4],
          labels[:n_train], labels[n_train:])
    plt.savefig('picture4/degree3.png', dpi=150)
    plt.close('all')

    # 实验三：前 10 列 = 9 阶 -> 预期过拟合：train 更低但 test 更高，weight 偏离真值
    print('--- 9 阶（10 列特征）---')
    train(poly_features[:n_train, :10], poly_features[n_train:, :10],
          labels[:n_train], labels[n_train:])
    plt.savefig('picture4/degree9.png', dpi=150)
    plt.close('all')

    print('三组跑完，图在 picture4/ 里：degree1.png、degree3.png、degree9.png')
    # 小提示：数据、噪声、初始化都是随机的，每次跑数字略有出入是正常现象，看走势和量级即可
