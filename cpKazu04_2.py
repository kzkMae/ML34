#coding:utf-8

import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''
df = pd.read_csv(StringIO(csv_data))

#############################################################################
print(50 * '=')
print('Section: Imputing missing values')
print(50 * '-')

# 欠測値補完のインスタンスを生成（平均値補完）
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)

#データを適合
imr = imr.fit(df)

#補完を実行
imputed_data = imr.transform(df.values)
print('Input Array:\n', df.values)
print('Imputed Data:\n', imputed_data)







