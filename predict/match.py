import pandas as pd

# 读取第一个 CSV 文件并按照 score 进行降序排序，选择前两百行
df1 = pd.read_csv('/home/suqun/tmp/GenScore-main/example/total5_unique2.csv')
df1_sorted = df1.sort_values(by='score', ascending=False).head(200)

# 读取第二个 CSV 文件
df2 = pd.read_csv('SuScore_MCL1_decoy.csv')

# 保留含有与第一个文件中相同 pdbid 的行
result = df2[df2['pdbid'].isin(df1_sorted['id'])]

# 将结果写入新的 CSV 文件
result.to_csv('matched_output.csv', index=False)

