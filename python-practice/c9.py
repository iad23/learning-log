def findMinAndMax(L):
    if isinstance(L, (list,tuple)):
        if not L:
            return (None, None)
        max=min=L[0]
        for x in L:
            if x>max:
                max=x
            if x<min:
                min=x
        return (min, max)
    return (None, None)
# 测试
if findMinAndMax([]) != (None, None):
    print('测试失败!')
elif findMinAndMax([7]) != (7, 7):
    print('测试失败!')
elif findMinAndMax([7, 1]) != (1, 7):
    print('测试失败!')
elif findMinAndMax([7, 1, 3, 9, 5]) != (1, 9):
    print('测试失败!')
else:
    print('测试成功!')