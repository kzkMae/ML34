#coding:utf-8

import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#############################################################################
print(50 * '=')
print('Section: First steps with scikit-learn')
print(50 * '-')

# Irisデータセットをロード
iris = datasets.load_iris()
# ３，４列目の特徴量を抽出
X = iris.data[:, [2, 3]]
#print(X)
# クラスラベルを取得
y = iris.target
#print(y)
#print('Class labels:', np.unique(y))

# トレーニングデータとテストデータに分割
# 全体の30％をテストデータにする
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
# トレーニングデータの平均と標準偏差を計算
sc.fit(X_train)
# 平均と標準偏差を用いて標準化
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


#############################################################################
"""
print(50 * '=')
print('Section: Training a perceptron via scikit-learn')
print(50 * '-')

# エポック数40，学習率0.1でパーセプトロンのインスタンスを生成
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
# トレーニングデータをモデルに適合させる
ppn.fit(X_train_std, y_train)

print('Y array shape', y_test.shape)

# テストデータで予測を実施
y_pred = ppn.predict(X_test_std)
# 誤分類のサンプルの個数を表示
print('Misclassified samples: %d' % (y_test != y_pred).sum())
# 分類の正解率を表示
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
"""






def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # マーカとカラーマップの準備
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
    # 予測結果を元のグリッドポイントのデータサイズに変更
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

    # highlight test samples
    """
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')
    """
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:,0], X_test[:,1],c='',
                       alpha=1.0, linewidths=1, marker='o', s=55, label='test set')


# トレーニングデータとテストデータの特徴量を行方向に結合
X_combined_std = np.vstack((X_train_std, X_test_std))
# トレーニングデータとテストデータのクラスラベルを結合
y_combined = np.hstack((y_train, y_test))
"""
# 決定領域のプロット
plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))
# 軸ラベルの設定
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
# 凡例の設定
plt.legend(loc='upper left')

# plt.tight_layout()
# plt.savefig('./figures/iris_perceptron_scikit.png', dpi=300)
# プロット
#plt.show()
plt.pause(5)
"""
#############################################################################
#############################################################################
'''
print(50 * '=')
print('Section: Training a logistic regression model with scikit-learn')
print(50 * '-')

# ロジスティック回帰のインスタンスを生成
lr = LogisticRegression(C=1000.0, random_state=0)
# トレーニングデータをモデルに適合させる
lr.fit(X_train_std, y_train)

# 決定境界をプロット
plot_decision_regions(X_combined_std, y_combined,
                      classifier=lr, test_idx=range(105, 150))
# 軸のラベルを設定
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
# 凡例の設定
plt.legend(loc='upper left')
# plt.tight_layout()
# plt.savefig('./figures/logistic_regression.png', dpi=300)
#plt.show()
# 表示
plt.show()
"""
print('Predicted probabilities', lr.predict_proba(X_test_std[0, :]
                                                  .reshape(1, -1)))
'''
#############################################################################
#############################################################################
'''
print(50 * '=')
print('Section: Tackling overfitting via regularization')
print(50 * '-')

# 空のリストを生成（重み係数，逆正則化パラメータ）
weights, params = [], []
# 10個の逆正則化パラメータに対応するロジスティック回帰モデルをそれぞれ処理
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10**c, random_state=0)
    lr.fit(X_train_std, y_train)
    # 重み係数を格納
    weights.append(lr.coef_[1])
    # 逆正則化パラメータを格納
    params.append(10**c)

# 重み係数をNumpy配列に変換
weights = np.array(weights)

plt.plot(params, weights[:, 0],
         label='petal length')
plt.plot(params, weights[:, 1], linestyle='--',
         label='petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
# 横軸を対数スケールに設定
plt.xscale('log')
# plt.savefig('./figures/regression_path.png', dpi=300)
plt.show()
'''
#############################################################################
#############################################################################
#'''
print(50 * '=')
print('Section: Dealing with the nonlinearly'
      'separable case using slack variables')
print(50 * '-')

# 線形SVMのインスタンスを生成
svm = SVC(kernel='linear', C=1.0, random_state=0)
# 線形SVMのモデルにトレーニングデータを適合させる
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
# plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_linear.png', dpi=300)
plt.show()
#'''
#############################################################################
# 確率的勾配降下法バージョンのパーセプトロンを生成
# ppn = SGDClassifier(loss='paeceptron')
# 確率的勾配降下法バージョンのロジスティック回帰を生成
# lr = SGDClassifier(loss='log')
# 確率的勾配降下法バージョンのSVM（損失関数=ヒンジ関数）を生成
# svm = SGDClassifier(loss='hinge')
#############################################################################
'''
print(50 * '=')
print('Section: Solving non-linear problems using a kernel SVM')
print(50 * '-')

# 乱数種を指定
np.random.seed(0)
# 標準正規分布に従う乱数で200行２列の行列を生成
X_xor = np.random.randn(200, 2)
# ２つの引数に対して排他的論理和を実行
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
# 排他的論理和の値が真の場合は１，魏の場合は-1を割り当てる
y_xor = np.where(y_xor, 1, -1)

# ラベル１を青のxでプロット
plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            c='b', marker='x',
            label='1')
# ラベル-1を赤の四角でプロット
plt.scatter(X_xor[y_xor == -1, 0],
            X_xor[y_xor == -1, 1],
            c='r',
            marker='s',
            label='-1')

# 軸の範囲を設定
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc='best')
# plt.tight_layout()
# plt.savefig('./figures/xor.png', dpi=300)
# プロット
plt.show()
'''
#############################################################################
#############################################################################
'''
print(50 * '=')
print('Section: Using the kernel trick to find separating hyperplanes'
      'in higher dimensional space')
print(50 * '-')

# RBFカーネルによるSVMのインスタンスを生成
svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor,
                      classifier=svm)

plt.legend(loc='upper left')
# plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_rbf_xor.png', dpi=300)
plt.show()

# RBFカーネルによるSVMのインスタンスを生成（２つのパラメータを生成）
svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
# plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_rbf_iris_1.png', dpi=300)
plt.show()

# RBFカーネルによるSVMのインスタンスを生成（γのパラメータを変更）
svm = SVC(kernel='rbf', random_state=0, gamma=100.0, C=1.0)
svm.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
# plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_rbf_iris_2.png', dpi=300)
plt.show()

'''
#############################################################################
#############################################################################
'''
print(50 * '=')
print('Section: Decision tree learning')
print(50 * '-')

# ジニ不純度の関数を定義
def gini(p):
    return p * (1 - p) + (1 - p) * (1 - (1 - p))

# エントロピーの関数を定義
def entropy(p):
    return - p * np.log2(p) - (1 - p) * np.log2((1 - p))

# 分類誤差関数を定義
def error(p):
    return 1 - np.max([p, 1 - p])

# 確率を表す配列を生成(0から0.99まで0.01刻み)
x = np.arange(0.0, 1.0, 0.01)

# 配列の値を基にエントロピー，分類誤差を計算
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e * 0.5 if e else None for e in ent]
err = [error(i) for i in x]

# 図の作成を開始
fig = plt.figure()
ax = plt.subplot(111)
# エントロピー（２種），ジニ不純度，分類誤差のそれぞれをループ処理
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
                          ['Entropy', 'Entropy (scaled)',
                           'Gini Impurity', 'Misclassification Error'],
                          ['-', '-', '--', '-.'],
                          ['black', 'lightgray', 'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

# 凡例の設定
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=3, fancybox=True, shadow=False)

# ２本の水平の破線を引く
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.ylabel('Impurity Index')
# plt.tight_layout()
# plt.savefig('./figures/impurity.png', dpi=300, bbox_inches='tight')
plt.show()
'''

#############################################################################
#############################################################################
'''
print(50 * '=')
print('Section: Building a decision tree')
print(50 * '-')

# エントロピーを指標とする決定木のインスタンスを生成
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
# 決定木のモデルにトレーニングデータを適合させる
tree.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined,
                      classifier=tree, test_idx=range(105, 150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
# plt.tight_layout()
# plt.savefig('./figures/decision_tree_decision.png', dpi=300)
plt.show()

# export_graphviz(tree,
#                out_file='tree.dot',
#                feature_names=['petal length', 'petal width'])
'''

#############################################################################
#############################################################################
'''
print(50 * '=')
print('Section: Combining weak to strong learners via random forests')
print(50 * '-')

# エントロピーを指標とするランダムフォレストのインスタンスを生成
forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=10,
                                random_state=1,
                                n_jobs=2)
# ランダムフォレストのモデルにトレーニングデータを適合させる
forest.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined, y_combined,
                      classifier=forest, test_idx=range(105, 150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
# plt.tight_layout()
# plt.savefig('./figures/random_forest.png', dpi=300)
plt.show()
'''

#############################################################################
#############################################################################
'''
print(50 * '=')
print('Section: K-nearest neighbors - a lazy learning algorithm')
print(50 * '-')

knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined_std, y_combined,
                      classifier=knn, test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
# plt.tight_layout()
# plt.savefig('./figures/k_nearest_neighbors.png', dpi=300)
plt.show()
'''



