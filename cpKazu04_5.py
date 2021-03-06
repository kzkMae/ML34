#coding:utf-8

from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

#############################################################################
print(50 * '=')
print('Section: Sequential feature selection algorithms')
print(50 * '-')


class SBS():
    """
    逐次後退選択（sequential backward selection）を実行するクラス
    """
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring                              # 特徴量を評価する指標
        self.estimator = clone(estimator)                   # 推定器
        self.k_features = k_features                        # 選択する特徴量の個数
        self.test_size = test_size                          # テストデータの割合
        self.random_state = random_state                    # 乱数種を固定する random_state

    def fit(self, X, y):
        # トレーニングデータとテストデータに分割

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        # すべての特徴量の個数，列インデックス
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        # すべての特徴量を用いてスコアを算出
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        # スコアを格納
        self.scores_ = [score]

        # 指定した特徴量の個数になるまで処理を反復
        while dim > self.k_features:
            # 空のリストの生成（スコア、列インデックス）
            scores = []
            subsets = []

            # 特徴量の部分集合を表す列インデックスの組み合わせごとに処理を反復
            for p in combinations(self.indices_, r=dim - 1):
                # スコアを算出して格納
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
                scores.append(score)
                # 特徴量の部分集合を表す列インデックスのリストを格納
                subsets.append(p)

            # 最良のスコアのインデックスを抽出
            best = np.argmax(scores)
            # 最良のスコアとなる列インデックスを抽出して格納
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            # 特徴量の個数を１つだけへらして次のステップへ
            dim -= 1

            # スコアを格納
            self.scores_.append(scores[best])
        # 最後に格納したスコア
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        # 抽出した特徴量を返す
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        # 指定された列番号indicesの特徴量を抽出してモデルに適合
        self.estimator.fit(X_train[:, indices], y_train)
        # テストデータを用いてクラスラベルを予測
        y_pred = self.estimator.predict(X_test[:, indices])
        # 秦のクラスラベルと予測値を用いてスコアを算出
        score = self.scoring(y_test, y_pred)
        return score


knn = KNeighborsClassifier(n_neighbors=2)

# selecting features
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
# plt.tight_layout()
# plt.savefig('./sbs.png', dpi=300)
plt.show()


k5 = list(sbs.subsets_[8])
print('Selected top 5 features:\n', df_wine.columns[1:][k5])

knn.fit(X_train_std, y_train)
print('\nPerformance using all features:\n')
print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))

knn.fit(X_train_std[:, k5], y_train)
print('\nPerformance using the top 5 features:\n')
print('Training accuracy:', knn.score(X_train_std[:, k5], y_train))
print('Test accuracy:', knn.score(X_test_std[:, k5], y_test))

#############################################################################
print(50 * '=')
print('Section: Assessing Feature Importances with Random Forests')
print(50 * '-')

feat_labels = df_wine.columns[1:]

forest = RandomForestClassifier(n_estimators=10000,
                                random_state=0,
                                n_jobs=-1)

forest.fit(X_train, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))

plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]),
        importances[indices],
        color='lightblue',
        align='center')

plt.xticks(range(X_train.shape[1]),
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
# plt.tight_layout()
# plt.savefig('./random_forest.png', dpi=300)
plt.show()

if Version(sklearn_version) < '0.18':
    X_selected = forest.transform(X_train, threshold=0.15)
else:
    from sklearn.feature_selection import SelectFromModel
    sfm = SelectFromModel(forest, threshold=0.15, prefit=True)
    X_selected = sfm.transform(X_train)

X_selected.shape
