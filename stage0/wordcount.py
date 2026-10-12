from collections import Counter
import string

# 第 1 步：读文件，把全部文字读成一个大字符串
with open("sample.txt", encoding="utf-8") as f:
    text = f.read()

# 第 2 步：统一转小写（The 和 the 算同一个词）
text = text.lower()

# 第 3 步：把标点都替换成空格（hello, 和 hello 算同一个词）
for p in string.punctuation:
    text = text.replace(p, " ")

# 第 4 步：按空格切开，变成单词列表
words = text.split()

# 第 5 步：数每个词出现几次
c = Counter(words)

# 第 6 步：打印 top-10
for word, count in c.most_common(10):
    print(f"{word}: {count}")
#top10 = sorted(c.items(), key=lambda x: x[1], reverse=True)[:10]
#for word, count in top10:
#   print(f"{word}: {count}")

