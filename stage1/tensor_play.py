import torch
x = torch.arange(12)
print(x)
print(x.shape)
print(x.size())
print(x.numel())
X=x.reshape(3,4)
print(X)
print(torch.zeros(2,3,4)) #形状为(2,3,4)的张量，其中所有元素都设置为0
print(torch.ones(2,3,4)) #形状为(2,3,4)的张量，其中所有元素都设置为1
print(torch.randn(3,4)) #每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样
print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x ** y,sep='\n')  # **运算符是求幂运算
print(torch.exp(x)) #求幂，底为e
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))  # (3,4)：[[0,1,2,3],[4,5,6,7],[8,9,10,11]]
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])  # (3,4)；因为有2.0，整体是浮点型
# dim=n：沿轴n拼接，被拼的轴变长，其他轴必须原本一样长
print(torch.cat((X, Y), dim=0))  # 沿轴0上下堆叠 → (6,4)：上3行X，下3行Y
print(torch.cat((X, Y), dim=1))  # 沿轴1左右拼接 → (3,8)：每行是X的4个数接Y的4个数
print(X == Y,X < Y,X > Y,sep='\n') #按元素逐个比较，返回布尔张量（形状不变）
print(X.sum()) #对张量中的所有元素进行求和，会产生一个单元素张量。
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a,b,sep='\n')
print(a+b) #形状不匹配时我们将两个矩阵广播为一个更大的矩阵，矩阵a将复制列，矩阵b将复制行，然后再按元素相加
print(X[-1], X[1:3],sep='\n') #用[-1]选择最后一个元素，可以用[1:3]选择第二个和第三个元素(左闭右开)
X[1, 2] = 9  # 索引赋值：把第1行第2列的元素改成9（原地修改，X本身变了）
print(X)     # 第1行变成 [4., 5., 9., 7.]
X[0:2, :] = 12  # 批量赋值：第0、1行（左闭右开）所有列都改成12
print(X)        # 前两行全变成12，第2行 [8., 9., 10., 11.] 不动
# ---- 原地操作：换水 vs 换杯子 ----
before = id(Y)
Y = Y + X                  # 换杯子：算出新结果，Y改指向新内存，id变了
print(id(Y) == before)     # False

Z = torch.zeros_like(Y)    # 建一个和Y同形状的全0张量
print('id(Z):', id(Z))
Z[:] = X + Y               # 换水：结果写进Z原来的内存，id不变
print('id(Z):', id(Z))     # 两次id一样

before = id(X)
X += Y                     # 原地加，X还是原来那个对象
print(id(X) == before)     # True

# ---- 转 NumPy ----
A = X.numpy()              # torch → numpy；A和X共享底层内存，改一个另一个也变
B = torch.tensor(A)        # numpy → torch；注意这是拷贝，想共享内存用 torch.from_numpy(A)
print(type(A), type(B))    # <class 'numpy.ndarray'> <class 'torch.Tensor'>

# ---- 取标量 ----
a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))  # tensor([3.5000]) 3.5 3.5 3
# .item()：把大小为1的张量转成Python标量
a = torch.arange(4).reshape((2, 1, 2))  # (2,1,2)
b = torch.arange(3).reshape((1, 3, 1))  # (1,3,1)
print(a + b)
# 形状：(2,1,2) vs (1,3,1) → (2,3,2)
# tensor([[[0, 1],
#          [1, 2],
#          [2, 3]],
#         [[2, 3],
#          [3, 4],
#          [4, 5]]])
#对齐看：最右维 2 vs 1 → 拉成 2；中间 1 vs 3 → 拉成 3；最左 2 vs 1 → 拉成 2。如果某维既不相等又不含 1（比如 (3,) 和 (2,)），就直接报错。
# ---- GPU 验证 ----
x = torch.rand(3, 4)
print(x.device)              # 应该是 cpu
y = x.to("cuda")             # 搬上 GPU
print(y.device)              # 应该是 cuda:0
print(torch.cuda.get_device_name(0))  # 打印显卡名字