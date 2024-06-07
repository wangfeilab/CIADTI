import numpy as np
with open('array_data6.txt', 'r') as f:
    S = [float(line.strip()) for line in f]
    count_of_ones = 0
    for element in S:
        if element == 1:
            count_of_ones += 1
    count = sum(1 for x in S if x > 0.5)
print(S)
print("列表S中大于0.5的元素个数为:", count)
print("数组中值为1的元素个数:", count_of_ones)
print(len(S))
sorted_indices = np.argsort(S)

# 取最后的 100 个索引（最大的 100 个数的索引）
top_100_indices = sorted_indices[-100:]

print("最大的100个数的索引：", top_100_indices)
import pandas as pd

# 读取CSV文件
df = pd.read_csv('../data/ins/HY-QS-1-column-3.csv', header = None)
print(df)
print(len(df))
# 假设您的下标数组是 indices

# 根据下标数组从CSV文件中找出对应的数据
selected_data = df.loc[top_100_indices]
print(selected_data)
selected_data.to_csv('selected_data1.csv', index=False)
