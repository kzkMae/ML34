#coding:utf-8

#機械学習用function格納

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold
from MachineLearning import *
import datetime

#Datasetをテストデータとトレーニングデータとテストデータに分割
def devisionDataset_testtrain(X, y, testsize):
    if testsize < 0.1 or testsize > 0.5:
        print('テストサイズは0.1～0.5以内に収めて')
        return 0,0,0,0,False
    print('分割サイズ（％）：', testsize * 100)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=0)
    return x_train, x_test, y_train, y_test, True

#標準化のインスタンスを生成
def datasetStandard(train_set, test_set):
    stdsc = StandardScaler()
    train_std = stdsc.fit_transform(train_set)
    test_std = stdsc.transform(test_set)
    return train_std, test_std

#k分割交差検証
def getKfold(y_train):
    print("交差検証用の数値を数値を入力してください\n k = ",end="")
    k = int(input())
    #print(k, type(k))
    return StratifiedKFold(y=y_train, n_folds=k, random_state=1)

#10分割交差検証
def get10fold(y_train):
    return StratifiedKFold(y=y_train, n_folds=10, random_state=1)

# 機械学習技術に投入用関数
def MLexecute(X_train, X_test, y_train, y_test, X_train_std, X_test_std, kfold=None):
    scores = {}
    times = {}
    #print('OKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK')
    #decisionTreeML(X_train, y_train, X_test, y_test, kfold)
    #testML(X_train, y_train)
    #testrandomForest(X_train,y_train, X_test, y_test)
    #testKNN(X_train_std,y_train, X_test_std, y_test)
    #testSVM(X_train_std, y_train, X_test_std, y_test)
    svmgs = gredsearchSVM(X_train_std, y_train, X_test_std, y_test)
    #print("欲しい情報 \n\n\n\n")
    dtgs = gredsearchDecisionTree(X_train, y_train, X_test, y_test)
    rfgs = gredsearchRandomForest(X_train, y_train, X_test, y_test)
    kngs = gredsearchKNN(X_train_std, y_train, X_test_std, y_test)
    scores, times, scores10 = testdatasetML10fold(X_train, y_train, X_train_std, X_test, y_test,X_test_std, svmGS=svmgs,treeGS=dtgs,forestGS=rfgs,knnGS=kngs)
    #testlearn(X_train,y_train)
    #testgredsearch(X_train, y_train, X_test, y_test)
    #gredsearchRandomForest(X_train, y_train, X_test, y_test)
    #gredsearchKNN(X_train_std, y_train, X_test_std, y_test)
    #gredsearchDecisionTree(X_train, y_train, X_test, y_test)
    return scores, times, scores10


def writeResult(columns, size, scores, times, scores10={}, trainsize=0,testsize=0):
    now = datetime.datetime.now()
    mllist = ['forest', 'knn', 'svm', 'tree']
    filename = 'result-{0:%Y_%m_%d__%H_%M_%S}.txt'.format(now)
    #print(filename)
    with open(filename, 'w') as f:
        f.write('特徴量の数：{}\n'.format(len(columns)) )
        f.write('特徴量：\n')
        for k, i in enumerate(columns):
            f.write('\t{0:2d}_{1}\n'.format(k,i))
        f.write('テストデータのサイズ：{}％\n'.format(size))
        f.write('トレーニングデータの数 ：{}検体\n'.format(trainsize))
        f.write('テストデータの数       ：{}検体\n'.format(testsize))
        f.write('10分割交差検証の分類精度\n')
        for i in mllist:
            f.write('{0:5s}:\n'.format(i))
            for k, j in enumerate(scores10[i]):
                #f.write('\t  {0:2d} : {1:.4f}\n'.format(k+1, j))
                f.write('\t  {0:.4f}\n'.format(j))
        for i in mllist:
            #print(type(i), type(scores[i]))
            f.write('{}_Accuracy\t\t:{}\n'.format(i, scores[i]))
        f.write('10分割交差検証実行時間\n')
        for i in mllist:
            f.write('\t{0:6s}:{1:.3f}s\n'.format(i, times[i]))
    return 0

