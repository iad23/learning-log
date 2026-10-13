import torch
print("=== 1. 基本流程：requires_grad + backward ===")
x = torch.arange(4.0)
x.requires_grad_(True) # 告诉 PyTorch：我要对 x 求导，给我记账,记下状态
#这两句等价于x = torch.arange(4.0,requires_grad=True)
#注意 requires_grad_ 末尾的下划线：表示原地操作

y = 2*torch.dot(x,x)  # y = 2*(x0²+x1²+x2²+x3²) = 28.0;torch.dot：两个一维向量的点积，结果是个标量。
print("y =", y) #此时 y 自带 grad_fn=<MulBackward0>：记账本，记着"y 是由乘法算出来的"
y.backward()
print("x.grad=",x.grad)
print("验证 x.grad == 4x:",x.grad == 4*x)

print("\n=== 2. 梯度累积：记得清零 ===")
#PyTorch 默认累积梯度：不清零，新梯度会加在旧梯度上，这就是为什么每次算之前要先写x.grad.zero_()
x.grad.zero_()
y = x.sum() #y=x0+x1+x2+x3
y.backward()
print("x.grad =",x.grad) #[1,1,1,1],如果不清零，这里会是 [1, 5, 9, 13]，错！

print("\n=== 3. 非标量 backward ===")
x.grad.zero_()
y = x*x
# y 是向量，不是标量(x*x是逐元素相乘)
# backward 要求起点是标量，所以先 y.sum() 变成标量再 backward
# 等价写法：y.backward(torch.ones(len(x)))
y.sum().backward() #先转化为标量
print("x.grad =", x.grad)

print("\n=== 4. detach：撕掉记账本 ===")
x.grad.zero_()
y=x*x # y = [0, 1, 4, 9]，y 记得自己是 x*x 算出来的
u = y.detach()  # u 数值和 y 一模一样，但"失忆"了：不记得自己咋来的,也就是u不知道自己是由x*x得来的
z = u*x  # z = [0, 1, 8, 27]
z.sum().backward()  # 梯度流到 u 就停了，u 被当成常数,因为u不知道自己怎么来的
print("x.grad == u:", x.grad == u) 
# 对比：如果不用 detach，z = x*x*x，梯度会是 3x²

print("\n=== 5. 控制流的梯度 ===")
def f(a):
    b = a*2
    while b.norm()<1000:  # norm = 向量长度；循环跑几圈取决于 a 的值，即向量的模
        b = b*2
    if b.sum()>0:
        c=b
    else:
        c = 100*b
    return c

a=torch.randn(size = (),requires_grad=True) # 标量随机数
d = f(a)
d.backward()
# 不管循环跑几圈、走哪个分支，d 总等于 k*a（k 是标量）
# 所以 d 对 a 的导数就是 k = d/a
print("a.grad == d / a:", a.grad == d / a)  # tensor(True)
