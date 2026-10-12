def trim(s):
    while s.startswith(' '):
        s = s[1:]
    while s.endswith(' '):
        s = s[:-1]
    return s
# 测试:
if trim('hello  ') != 'hello':
    print('1测试失败!')
elif trim('  hello') != 'hello':
    print('2测试失败!')
elif trim('  hello  ') != 'hello':
    print('3测试失败!')
elif trim('  hello  world  ') != 'hello  world':
    print('4测试失败!')
elif trim('') != '':
    print('5测试失败!')
elif trim('    ') != '':
    print('6测试失败!')
else:
    print('测试成功!')