height = float(input('请输入身高（米）：'))
weight = float(input('请输入体重（公斤）：'))
bmi = weight / (height ** 2)
if bmi < 18.5:
    print('过轻')
elif bmi < 25:
    print('正常')
elif bmi < 28:
    print('过重')
else:
    print('肥胖')