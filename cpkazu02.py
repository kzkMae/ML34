#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

class Perceptron(object):
    """
    パーセプトロンの分類機

    パラメータ
    ----------------------------------
    eta : float
        学習率（0.0 <= x <= 1.0）
    n_iter : int
        トレーニングデータのトレーニング回数


    属性
    ----------------------------------
    w_ : 1次元配列
        適合後の重み
    errors_ : リスト
        各エポックでの誤分類数
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter


    def fit(self, X, y):
        """
        トレーニングデータに適合させる

        パラメータ
        -----------------------------
        X : {配列のようなデータ構造}, shape = [n_sample, n_features]
            トレーニングデータ
            n_sampleはサンプルの個数，n_featuresは特徴量の個数
        y : 配列のようなデータ構造, shape = [n_sample]
            目的変数


        戻り値
        -------------------------------
        self : object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):                                #トレーニング回数分トレーニングデータを反復
            errors = 0
            for xi, target in zip(X, y):                            #各サンプルで重みを更新
                #重み w1,...,wmの更新
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                # 重み w0の更新
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """総入力を計算"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """１ステップ後のクラスラベルを返す"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


#############################################################################
print(50 * '=')
print('Section: Training a perceptron model on the Iris dataset')
print(50 * '-')

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases/iris/iris.data', header=None)
#print(df.tail())

#############################################################################
print(50 * '=')
print('Plotting the Iris data')
print(50 * '-')

# １～100行目の目的変数を抽出
y = df.iloc[0:100, 4].values
# Setosaを-1，Virginicaを１に変換
y = np.where(y == 'Iris-setosa', -1, 1)

# 1-100行目の１，３列目を抽出
X = df.iloc[0:100, [0, 2]].values

# Setosaのプロット（赤○）
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
# Versicolorプロット（青×）
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

#軸ラベルの設定
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
#凡例の設定
plt.legend(loc='upper left')

# plt.tight_layout()
# plt.savefig('./images/02_06.png', dpi=300)
#表示
plt.show()

#############################################################################
#############################################################################
print(50 * '=')
print('Training the perceptron model')
print(50 * '-')

# パーセプロトロンのオブジェクトを生成
ppn = Perceptron(eta=0.1, n_iter=10)

# トレーニングデータへのモデル適合
ppn.fit(X, y)

# エポックと誤分類誤差の関係の折れ線グラフをプロット
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
# 軸ラベルの設定
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')

# plt.tight_layout()
# plt.savefig('./perceptron_1.png', dpi=300)
# プロット
plt.show()

#############################################################################

print(50 * '=')
print('A function for plotting decision regions')
print(50 * '-')


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # マーカーとカラーマップの準備
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 決定領域のプロット
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # グリッドポイントの生成
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # 各特徴量を１次元配列に変換して予測を実行
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # 予測結果を基のグリッドポイントのデータサイズに変更
    Z = Z.reshape(xx1.shape)
    # グリッドポイントの等高線のプロット
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    # 軸の範囲の設定
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # クラスごとにサンプルをプロット
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

# 決定境界のプロット
plot_decision_regions(X, y, classifier=ppn)
# 軸ラベルの設定
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
# 凡例の設定
plt.legend(loc='upper left')

# plt.tight_layout()
# plt.savefig('./perceptron_2.png', dpi=300)
# 表示
plt.show()


#############################################################################
#############################################################################
print(50 * '=')
print('Implementing an adaptive linear neuron in Python')
print(50 * '-')


class AdalineGD(object):
    """
    ADAptive LInear NEuron分類機

    パラメータ
    ------------
    eta : float
        学習率（0.0 < x <= 1.0）
    n_iter : int
        トレーニングデータのトレーニング回数

    属性
    -----------
    w_ : １次元配列
        適合後の重み
    errors_ : list
        各エポックでの誤分類数

    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        トレーニングデータに適合させる

        パラメータ
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            トレーニングデータ
            n_samples はサンプルの個数
            n_featuresは特徴量の個数
        y : array-like, shape = [n_samples]
            目的変数

        戻り値
        -------
        self : object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):    # トレーニング回数文トレーニングデータを反復
            # 活性化関数の出力の計算
            output = self.net_input(X)
            # 誤差の計算
            errors = (y - output)
            # 重みの更新
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            # コスト関数の計算
            cost = (errors**2).sum() / 2.0
            # コストの格納
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """総入力を計算"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """線形活性化関数の出力を計算"""
        return self.net_input(X)

    def predict(self, X):
        """１ステップ後のクラスラベルを返す"""
        return np.where(self.activation(X) >= 0.0, 1, -1)

# 描画領域を１行２列に分割
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

# 勾配降下法によるADLINEの学習(学習率 eta=0.01)
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
# エポック数とコストの関係を表す折れ線グラフのプロット
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
# 軸のラベルの設定
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
# タイトルの設定
ax[0].set_title('Adaline - Learning rate 0.01')

# 勾配降下法によるADLINEの学習(学習率 eta=0.0001)
ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
# エポック数とコストの関係を表す折れ線グラフのプロット
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
# 軸のラベルの設定
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
# タイトルの設定
ax[1].set_title('Adaline - Learning rate 0.0001')

# plt.tight_layout()
# plt.savefig('./adaline_1.png', dpi=300)
plt.show()


print('standardize features')
# データのコピー
X_std = np.copy(X)
# 各列の標準化
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

# 勾配降下法によるADLINEの学習(標準化後、学習率 eta=0.01)
ada = AdalineGD(n_iter=15, eta=0.01)
# モデルの適合
ada.fit(X_std, y)

# 境界領域のプロット
plot_decision_regions(X_std, y, classifier=ada)
# タイトルの設定
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
# 凡例の設定
plt.legend(loc='upper left')
# plt.tight_layout()
# plt.savefig('./adaline_2.png', dpi=300)
# 表示
plt.show()

# エポック数とコストの関係を表す折れ線グラフのプロット
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')

# plt.tight_layout()
# plt.savefig('./adaline_3.png', dpi=300)
# 図の表示
plt.show()


#############################################################################
#############################################################################
print(50 * '=')
print('Large scale machine learning and stochastic gradient descent')
print(50 * '-')


class AdalineSGD(object):
    """
    ADAptive LInear NEuron分類機

    パラメータ
    ------------
    eta : float
        学習率 (0.0 < eta <= 1.0)
    n_iter : int
        トレーニングデータのトレーニング回数

    属性
    -----------
    w_ : 1次元配列
        適合後の重み
    errors_ : list
        各エポックでの誤分類数
    shuffle : bool (default: True)
        巡回を回避するために各エポックでトレーニングデータをシャッフル
    random_state : int (default: None)
        シャッフルに使用するrandom_stateを設定し，重みを初期化

    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        # 学習率の初期化
        self.eta = eta
        # トレーニング回数の初期化
        self.n_iter = n_iter
        # 重みの初期化フラグはFalseに設定
        self.w_initialized = False
        # 各エポックでトレーニングデータをシャッフルするかどうかのフラグの初期化
        self.shuffle = shuffle
        # 引数random_stateが指定された場合は乱数種を設定
        if random_state:
            np.random.seed(random_state)

    def fit(self, X, y):
        """
         トレーニングデータに適合させる

        パラメータ
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            トレーニングデータ
            n_samples is サンプル数
            n_features is 特徴量の個数
        y : array-like, shape = [n_samples]
            目的変数

        戻り値
        -------
        self : object

        """
        # 重みベクトルの生成
        self._initialize_weights(X.shape[1])
        # コストを格納するリストの生成
        self.cost_ = []
        # トレーニング回数分トレーニングデータを反復
        for i in range(self.n_iter):
            # 指定された場合はトレーニングデータをシャッフル
            if self.shuffle:
                X, y = self._shuffle(X, y)
            # 各サンプルのコストを格納するリストを生成
            cost = []
            # 各サンプルに対する計算
            for xi, target in zip(X, y):
                # 特徴量xiと目的変数yを用いた重みの更新とコストの計算
                cost.append(self._update_weights(xi, target))
            # サンプルの平均コストの計算
            avg_cost = sum(cost) / len(y)
            # 平均コストの格納
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """重みを再初期化することなくトレーニングデータに適合させる"""
        # 初期化されていない場合は初期化を実行
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        # 目的変数yの要素数が２以上の場合は，各サンプルの特徴量xiと目的変数targetで重みを更新
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        # 目的変数yの要素数が１の場合はサンプル全体の特徴量Xと目的変数yで重みを更新
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """トレーニングデータをシャッフル"""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """重みを零に初期化"""
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """ADALINEの学習規則を用いて重みを更新"""
        # 活性化関数の出力の計算
        output = self.net_input(xi)
        # 誤差の計算
        error = (target - output)
        # 重みの更新
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        # コストの計算
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        """総入力の計算"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """線形活性化関数の出力を計算"""
        return self.net_input(X)

    def predict(self, X):
        """１ステップ後のクラスラベルを返す"""
        return np.where(self.activation(X) >= 0.0, 1, -1)


# 確率的勾配降下法によるADLINEの学習
ada = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
# モデルへの適合
ada.fit(X_std, y)

# 境界領域をプロット
plot_decision_regions(X_std, y, classifier=ada)
# タイトルの設定
plt.title('Adaline - Stochastic Gradient Descent')
# 軸ラベルの設定
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
# 凡例の設定
plt.legend(loc='upper left')

plt.tight_layout()
# plt.savefig('./adaline_4.png', dpi=300)
# プロットの表示
plt.show()

# エポックとコストの折れ線グラフのプロット
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')

# plt.tight_layout()
# plt.savefig('./adaline_5.png', dpi=300)
# プロット
plt.show()




