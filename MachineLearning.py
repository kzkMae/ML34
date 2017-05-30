#coding:utf-8

#--Features 0 1 2 3 4 5 6 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38

#機械学習技術実装関数

from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
import time

#決定木
def testdecisionTree(x_train,y_train,x_test, y_test,kfold):
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=3,random_state=0)
    #tree.fit(x_train, y_train)
    for k, (train, test) in enumerate(kfold):
        tree.fit(x_train[train], y_train[train])
        score = tree.score(x_train[test], y_train[test])
        print('Fold: %2s, Class dist.: %s, ACC: %.3f' % (k+1, np.bincount(y_train[train]), score))
    #tree.score()

def testML(X_train, y_train):
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=1000, random_state=0)
    scores = cross_val_score(estimator=tree,X=X_train, y=y_train, cv=10, n_jobs=1)
    print('CV accuracy scores: %s' % scores)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    pipe_lr = Pipeline([('scl', StandardScaler()),('pca', PCA(n_components=32)),('tree',tree)])
    print('-----------------------------------------------------------------------------------------')
    scores = cross_val_score(estimator=pipe_lr,X=X_train, y=y_train, cv=10, n_jobs=1)
    print('CV accuracy scores: %s' % scores)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

def testrandomForest(X_train, y_train,X_test, y_test):
    forest = RandomForestClassifier(criterion='entropy', n_estimators=1000, random_state=1, n_jobs=-1)
    forest.fit(X_train, y_train)
    importance = forest.feature_importances_
    indices = np.argsort(importance)[::-1]
    for f in range(X_train.shape[1]):
        print('%2d) %-*s %f' % (f+1 , 30, indices[f], importance[indices[f]]))
    #print('acc : %s' % forest)
    testreg = forest.predict(X_test)
    #print(classification_report(y_test, testreg))
    print(accuracy_score(y_test, testreg))

def testKNN(X_train_std, y_train, X_test_std, y_test):
    knn = KNeighborsClassifier(n_neighbors = 5, p=2, metric='minkowski')
    knn.fit(X_train_std, y_train)
    print('score: %s' % knn.score(X_test_std, y_test))

def testSVM(X_train_std, y_train, X_test_std, y_test):
    svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
    svm.fit(X_train_std, y_train)
    test_pred = svm.predict(X_test_std)
    print(classification_report(y_test, test_pred))
    print(accuracy_score(y_test, test_pred))
    return 0

# 機械学習により，トレーニングデータを投入
#10分割交差検証による評価だけを抜き出す
def testdatasetML10fold(X_train, y_train, X_train_std, X_test, y_test, X_test_std, svmGS):
    scores = {}
    scores10 = {}
    times ={}
    #決定木
    tree = DecisionTreeClassifier(criterion='entropy',max_depth=10,random_state=0)
    #ランダムフォレスト
    forest = RandomForestClassifier(criterion='entropy', n_estimators=10,random_state=1)
    #KNN
    knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    #SVM
    #svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
    svm = svmGS
    #各々のスコアを格納
    starts = time.time()
    tree_scores = cross_val_score(estimator=tree, X=X_train, y=y_train, cv=10, n_jobs=1)
    ends = time.time()
    times['tree'] = (ends - starts)
    starts = time.time()
    forest_scores = cross_val_score(estimator=forest, X=X_train, y=y_train, cv=10, n_jobs=1)
    ends = time.time()
    times['forest'] = (ends - starts)
    starts = time.time()
    knn_scores = cross_val_score(estimator=knn, X=X_train_std, y=y_train, cv=10, n_jobs=1)
    ends = time.time()
    times['knn'] = (ends - starts)
    starts = time.time()
    svm_scores = cross_val_score(estimator=svm, X=X_train_std, y=y_train, cv=10, n_jobs=1)
    ends = time.time()
    times['svm'] = (ends - starts)
    #結果を出力
    #print('tree\taccuracy: %.3f +/- %.3f' % (np.mean(tree_scores), np.std(tree_scores)))
    scores['tree'] = '{0:.3f} +/- {1:.3f}'.format(np.mean(tree_scores), np.std(tree_scores))
    scores10['tree'] = tree_scores
    #print('tree     : %s' % tree_scores)
    #print('forest\taccuracy: %.3f +/- %.3f' % (np.mean(forest_scores), np.std(forest_scores)))
    scores['forest'] = '{0:.3f} +/- {1:.3f}'.format(np.mean(forest_scores), np.std(forest_scores))
    scores10['forest'] = forest_scores
    #print('forest   : %s' % forest_scores)
    #print('knn\t\taccuracy: %.3f +/- %.3f' % (np.mean(knn_scores), np.std(knn_scores)))
    scores['knn'] = '{0:.3f} +/- {1:.3f}'.format(np.mean(knn_scores), np.std(knn_scores))
    scores10['knn'] = knn_scores
    #print('knn      : %s' % knn_scores)
    #print('svm\t\taccuracy: %.3f +/- %.3f' % (np.mean(svm_scores), np.std(svm_scores)))
    scores['svm'] = '{0:.3f} +/- {1:.3f}'.format(np.mean(svm_scores), np.std(svm_scores))
    scores10['svm'] = svm_scores
    #print('svm      : %s' % svm_scores)
    #print(scores)
    return scores, times, scores10

def testlearn(X_train, y_train):
    pipe_lr = Pipeline([('scl',StandardScaler()),('svm',SVC(kernel='rbf',random_state=0,gamma=0.2,C=1.0))])
    train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, X=X_train, y=y_train,train_sizes=np.linspace(0.1,1.0,10),cv=10)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores,axis=1)
    test_mean = np.mean(test_scores,axis=1)
    test_std = np.std(test_scores,axis=1)

    plt.plot(train_sizes,train_mean,color='blue',marker='o', markersize=5,label='training accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha = 0.15, color = 'blue')

    plt.plot(train_sizes, test_mean, color='red',linestyle='--', marker='s', markersize=5, label='validation accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='red')
    plt.grid()

    plt.show()

#ハイパーパラメータ調整可能SVM
def gredsearchSVM(X_train_std, y_train, X_test_std,y_test):
    print('SVMパラメータ調整中')
    #pip_svc = Pipeline([('scl',StandardScaler()), ('cl',SVC(random_state=1))])
    svm = SVC(random_state=1)
    param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_range2 = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    param_degree = [3,4,5,7,9,10,100]
    #liner:線形，rbf：ガウス，poly：多項式，sigmoid：シグモイド，precomoyted：プレコンピューティッド
    '''param_grid = [{'C':param_range, 'gamma':param_range, 'kernel':['rbf']},
                  {'C':param_range, 'kernel':['linear']},
                  {'C':param_range, 'gamma':param_range, 'coef0':[100.0], 'kernel': ['poly']},
                  {'C':param_range, 'gamma':param_range, 'coef0':param_range2, 'kernel': ['sigmoid']}]'''
    #print('a')
    #最適パラメータ
    param_grid = [{'C': [100.0], 'gamma': [0.001], 'kernel': ['rbf']}]
    #               {'C': [1.0], 'kernel': ['linear']}]
    gs = GridSearchCV(estimator=svm,param_grid=param_grid,scoring='accuracy', cv=10)
    #print('b')
    gs = gs.fit(X_train_std,y_train)
    print('SVMパラメータ調整終了')
    print('best score: %.3f' % gs.best_score_)
    print(gs.best_params_)
    clf = gs.best_estimator_
    clf.fit(X_train_std, y_train)
    print('test accuracy: %.3f' % clf.score(X_test_std,y_test))
    print(gs.grid_scores_)
    #最優秀のパラメータを持つものを返却
    return gs.best_estimator_

def gredsearchDecisionTree(X_train, y_train, X_test, y_test):
    #pip_tree = Pipeline([('tree', DecisionTreeClassifier(criterion='entropy', random_state=0))])
    #調整するパラメータ
    #分割時の品質，分割を行う方式，使用する特徴数の最大値，深さ，最小値，
    tree = DecisionTreeClassifier(random_state=0)
    # 分割時の品質
    param_criter = ['entropy', 'gini']
    # 分割時の方法
    param_splitter = ['best','random']
    # 特徴数の最大数
    param_max_feature = ['auto', 'sqrt', 'log2', None]
    param_range = [1,2,3,4,5,6,7,8,9,10,20,30,40,100, None]
    # min_samples_split
    param_range2 = [2,4,6,8,10]
    # min_weight_fraction_leaf
    param_range3 = [0.0,0.001, 0.01, 0.1]
    # class_weight
    param_class_wight = [None, 'balanced']
    # criterion, splitter,max_features, max_depth
    param_grid = [{'criterion':param_criter, 'max_depth': param_range, 'splitter' : param_splitter, 'max_features' : param_max_feature,
                   'min_samples_split' : param_range2,'min_weight_fraction_leaf' : param_range3, 'class_weight' : param_class_wight}]
    gs = GridSearchCV(estimator=tree, param_grid= param_grid, scoring='accuracy', cv=10)
    gs.fit(X_train, y_train)
    print('best score: %.3f' % gs.best_score_)
    print(gs.best_params_)
    treegs = gs.best_estimator_
    treegs.fit(X_train,y_train)
    #print('test accuracy: %.3f' % treegs.score(X_test, y_test))
    #print(gs.grid_scores_)
    #最優秀のパラメータを持つものを返却
    return gs.best_estimator_


def gredsearchRandomForest(X_train, y_train, X_test, y_test):
    #pip_forest = Pipeline([('forest', RandomForestClassifier(criterion='entropy',random_state=1))])
    forest = RandomForestClassifier(random_state=1)
    param_range = [8,9,10,20,30,40,50,60,70,80,90,100,200,300]
    param_range2 = [1,5,10,15,20,50,100,None]
    # 分割時の品質
    param_criterion = ['entropy', 'gini']
    #特徴の最大値
    param_max_feature = ["auto","sqrt","log2",None,1,2,3,4,5,6,7,8,9]
    #重みづけ
    param_min_weight_fraction_leaf = [0.0,0.001,0.01,0.1,0.2,0.3,0.4]
    param_grid = [{'n_estimators': param_range, 'criterion' : param_criterion, 'max_features': param_max_feature,
                   'max_depth' :  param_range2, 'min_weight_fraction_leaf' : param_min_weight_fraction_leaf}]
    gs = GridSearchCV(estimator=forest, param_grid= param_grid, scoring='accuracy', cv=10)
    gs.fit(X_train, y_train)
    print('best score: %.3f' % gs.best_score_)
    print(gs.best_params_)
    forestgs = gs.best_estimator_
    forestgs.fit(X_train,y_train)
    print('test accuracy: %.3f' % forestgs.score(X_test, y_test))

def gredsearchKNN(X_train_std, y_train, X_test_std, y_test):
    #pip_knn = Pipeline([('knn', KNeighborsClassifier(metric='minkowski'))])
    knn = KNeighborsClassifier(metric='minkowski')
    param_range = [1,2,3,4,5,6,7,8,9,10]
    param_grid = [{'n_neighbors': param_range, 'p':param_range}]
    gs = GridSearchCV(estimator=knn, param_grid=param_grid, scoring='accuracy', cv=10)
    gs.fit(X_train_std, y_train)
    print('best score: %.3f' % gs.best_score_)
    print(gs.best_params_)
    forestgs = gs.best_estimator_
    forestgs.fit(X_train_std, y_train)
    print('test accuracy: %.3f' % forestgs.score(X_test_std, y_test))

#def testdatasetgredsearch(X_train, y_train, X_test, y_test):







