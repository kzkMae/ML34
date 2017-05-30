#coding:utf-8

import pandas as pd
from io import StringIO

'''
#############################################################################
print(50 * '=')
print('Section: Dealing with missing data')
print(50 * '-')
'''
# サンプルデータを作成
csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

# If you are using Python 2.7, you need
# to convert the string to unicode:
# csv_data = unicode(csv_data)

# サンプルデータを読み込む
df = pd.read_csv(StringIO(csv_data))
#print(df)
#print('\n\nExecuting df.isnull().sum():')
# 各特徴量の欠測値をカウント
#print(df.isnull().sum())

# NaNかどうかをTFで判断
#print(df.isnull())
#print(df.values)

#############################################################################
#############################################################################
print(50 * '=')
print('Section: Eliminating samples or features with missing values')
print(50 * '-')

'''print('\n\nExecuting df.dropna()')
# 欠測値を含む行を削除
print(df.dropna())'''

'''print('\n\nExecuting df.dropna(axis=1)')
# 欠測値を含む列を削除
print(df.dropna(axis=1))'''

'''print("\n\nExecuting df.dropna(thresh=4)")
print("(drop rows that have not at least 4 non-NaN values)")
# 非NaN値が４未満の行を削除
print(df.dropna(thresh=4))'''

'''print("\n\nExecuting df.dropna(how='all')")
print("(only drop rows where all columns are NaN)")
# 全ての列がNaNである行だけを削除
print(df.dropna(how='all'))'''

print("\n\nExecuting df.dropna(subset=['C'])")
print("(only drop rows where NaN appear in specific columns (here: 'C'))")
# 特定の列（例ではC）にNaNが含まれている行だけを削除
print(df.dropna(subset=['C']))


#############################################################################
