import math
import cmath

def quadratic(a, b, c):
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or not isinstance(c, (int, float)):
        raise TypeError('bad operand type')
    if a == 0:
        raise ValueError("a不能为0，不是一元二次方程")

    delta = b**2 - 4*a*c
    
    match delta:
        case d if d > 1e-10:
            s = math.sqrt(d)
            x1 = (-b + s) / (2*a)
            x2 = (-b - s) / (2*a)
            return(x1, x2)
        case d if abs(d) <= 1e-10:
            x1 = -b / (2*a)
            return (x1,)  #(x1,)代表返回的是单元素元组
        case d if d < -1e-10:
            s = cmath.sqrt(d)
            x1 = (-b + s) / (2*a)
            x2 = (-b - s) / (2*a)
            return(x1, x2)
# 返回值是tuple
print('quadratic(2, 3, 1) =', quadratic(2, 3, 1))
print('quadratic(1, 3, -4) =', quadratic(1, 3, -4))

if quadratic(2, 3, 1) != (-0.5, -1.0):
    print('测试失败')
elif quadratic(1, 3, -4) != (1.0, -4.0):
    print('测试失败')
else:
    print('测试成功')

# 新增重根测试
print('quadratic(1,2,1)=', quadratic(1,2,1))