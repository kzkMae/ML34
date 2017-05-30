#coding:utf-8

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#############################################################################
'''print(50 * '=')
print('Section: Partitioning a dataset in training and test sets')
print(50 * '-')'''

# Wineデータセットを読み込む
df_wine = pd.read_csv('./book/datasets/wine/wine.data',
                      header=None)

# 列名を設定
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

# クラスラベルを表示
#print('Class labels', np.unique(df_wine['Class label']))

# Wineデータセットの先頭５行を表示
#print('\nWine data excerpt:\n\n', df_wine.head())

# 特徴量とクラスラベルを別々に抽出
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
#print(y)


# トレーニングデータとテストデータに分割（全体の30％をテストデータ）
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0)


#############################################################################
#############################################################################
'''print(50 * '=')
print('Section: Bringing features onto the same scale')
print(50 * '-')'''


# min-maxスケーリングのインスタンスを生成
#mms = MinMaxScaler()
# トレーニングデータをスケーリング
#X_train_norm = mms.fit_transform(X_train)
# テストデータをスケーリング
#X_test_norm = mms.transform(X_test)

# 標準化のインスタンスを生成（平均=0, 標準偏差=1に変換）
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

ex = pd.DataFrame([0, 1, 2, 3, 4, 5])
#print('Scaling Example:\n')
#print('\nInput array:\n', ex)
ex[1] = (ex[0] - ex[0].mean()) / ex[0].std(ddof=0)

# Please note that pandas uses ddof=1 (sample standard deviation)
# by default, whereas NumPy's std method and the StandardScaler
# uses ddof=0 (population standard deviation)

# normalize
ex[2] = (ex[0] - ex[0].min()) / (ex[0].max() - ex[0].min())
ex.columns = ['input', 'standardized', 'normalized']
#print('\nOutput array after scaling:\n', ex)


#############################################################################
#############################################################################
print(50 * '=')
print('Section: Sparse solutions with L1-regularization')
print(50 * '-')

# L1正則化ロジスティック回帰のインスタンスを生成（逆正則化パラメータ C=0.1）
lr = LogisticRegression(penalty='l1', C=0.1)
# トレーニングデータに適合
lr.fit(X_train_std, y_train)
# トレーニングデータに対する正解率を表示
print('Training accuracy:', lr.score(X_train_std, y_train))
# テストデータに対する正解率の表示
print('Test accuracy:', lr.score(X_test_std, y_test))

# 切片の表示
print('Intercept:', lr.intercept_)
print('Model weights:', lr.coef_)

# 描画の準備
fig = plt.figure()
ax = plt.subplot(111)

# 各係数の色のリスト
colors = ['blue', 'green', 'red', 'cyan',
          'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue',
          'gray', 'indigo', 'orange']

# 空のリストを生成（重み係数，逆正則化パラメータ）
weights, params = [], []
for c in np.arange(-4, 6):
    lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10**c)

# 重み係数をNumpy配列に変換
weights = np.array(weights)

# 各重み係数をプロット
for column, color in zip(range(weights.shape[1]), colors):
    # 横軸を逆正則化パラメータ，縦軸を重み係数とした折れ線グラフ
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column + 1],
             color=color)
# y = 0に黒い破線を引く
plt.axhline(0, color='black', linestyle='--', linewidth=3)
# 横軸の範囲の設定
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',
          bbox_to_anchor=(1.38, 1.03),
          ncol=1, fancybox=True)
# plt.savefig('./figures/l1_path.png', dpi=300)
plt.show()


#############################################################################
