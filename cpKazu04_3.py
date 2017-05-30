#coding:utf-8

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import OneHotEncoder

#############################################################################
'''print(50 * '=')
print('Section: Handling categorical data')
print(50 * '-')'''

# サンプルデータを生成（Tシャツの色・サイズ・価格・クラスラベル）
df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                   ['red', 'L', 13.5, 'class2'],
                   ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']
#print('Input Array:\n', df)

#############################################################################
'''print(50 * '=')
print('Section: Mapping ordinal features')
print(50 * '-')'''

# Tシャツのサイズと整数を対応させるディクショナリを生成
size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}

# Tシャツのサイズを整数に変換
df['size'] = df['size'].map(size_mapping)
#print('Mapping:\n', df)

#inv_size_mapping = {v: k for k, v in size_mapping.items()}
#df_inv = df['size'].map(inv_size_mapping)
#print('\nInverse mapping:\n', df_inv)


#############################################################################
#############################################################################
'''print(50 * '=')
print('Section: Encoding class labels')
print(50 * '-')'''

# クラスラベルと整数値を対応させるディクショナリを生成
class_mapping = {label: idx for idx, label
                 in enumerate(np.unique(df['classlabel']))}
#print('\nClass mapping:\n', class_mapping)

# クラスラベルを整数に変換
df['classlabel'] = df['classlabel'].map(class_mapping)
#print('Mapping:\n', df)

# 整数とクラスラベルを対応させるディクショナリを生成
inv_class_mapping = {v: k for k, v in class_mapping.items()}
# 整数からクラスラベルに変換

df_inv = df['classlabel'] = df['classlabel'].map(inv_class_mapping)
#print('\nInverse mapping:\n', df_inv)

'''# ラベルエンコーダのインスタンスを生成
class_le = LabelEncoder()
# クラスらべrから整数を変換
y = class_le.fit_transform(df['classlabel'].values)
print('Label encoder tansform:\n', y)

# クラスラベルを文字列に戻す
y_inv = class_le.inverse_transform(y)
print('Label encoder inverse tansform:\n', y_inv)'''


#############################################################################
#############################################################################
print(50 * '=')
print('Section: Performing one hot encoding on nominal features')
print(50 * '-')

X = df[['color', 'size', 'price']].values

color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print("Input array:\n", X)

# one-hotエンコーダの生成
ohe = OneHotEncoder(categorical_features=[0])
# one-hotエンコーディングを実行
X_onehot = ohe.fit_transform(X).toarray()
print("Encoded array:\n", X_onehot)

# one-hotエンコーディングを実行
df_dummies = pd.get_dummies(df[['price', 'color', 'size']])
print("Pandas get_dummies alternative:\n", df_dummies)


#############################################################################




