#coding:utf-8

#import
import pandas as pd
import numpy as np
import argparse
from Function import *
from FunctionML import *

#引数や-hのオプションを定義
parser = argparse.ArgumentParser(prog='ML用スクリプト',description='オプションと引数の説明',
                                epilog='以上')
parser.add_argument('-v','--version', action='version', version='%(prog)s version')
parser.add_argument('DatasetFile',type=str, help='Datasetファイルのパスを指定, 型：%(type)s，String')
parser.add_argument('--Features',type=int, nargs='*', help='使用する特徴量を整数型で選択, 型：%(type)s，Int')
parser.add_argument('testSize',type=float, help='データセット分割サイズ（0.1～0.5の間）, 型：%(type)s，float')


# 引数を格納
arguMain = parser.parse_args()

# エラーチェック
errorCheck = True

# Datasetファイルの存在を確認
dataFile = arguMain.DatasetFile
errorCheck = isPath(dataFile)
deadErrorEnd(errorCheck)


#データセットを取得（読み込み）
# 0行目を列名に設定
#datasetMain = pd.read_csv(dataFile,header=0)
dataset = pd.read_csv(dataFile,header=0)
#Nameを除外
#dataset = datasetMain.iloc[:,1:]

# 特徴量の列を取得
featureslist = arguMain.Features
errorCheck = checkFeatureslist(len(dataset.columns)-2, featureslist)
deadErrorEnd(errorCheck)

#データセット分割サイズ
traintest_size = arguMain.testSize

#print(dataset.head())
#print(dataset.columns)
#print(len(dataset.columns))

#Labelのマッピングを作成，実行
labels = np.unique(dataset['Label'])
#print(labels)
label_mapping = {label:x for x, label in enumerate(labels)}
#print(label_mapping)
dataset['Label'] = dataset['Label'].map(label_mapping)
#print(dataset.head())

#特徴量を選択
#X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
if featureslist is None:
    Xpd, y = dataset.iloc[:, :-1], dataset.iloc[:, -1].values
    printFeatures(Xpd.columns)
else:
    Xpd, y = dataset.iloc[:, featureslist], dataset.iloc[:, -1].values
    printFeatures(Xpd.columns)
'''#t = 1,2,3
#d = [1,2,3]
#X = dataset.iloc[:, d]
#print('-----------------------------------------------')
#print(X.head())
#printFeatures(X.columns)
#print('***********************************************')
#print(y.head())
#print(dataset.columns[featureslist])
#one-hotエンコーディングを実行'''
Xone = pd.get_dummies(Xpd)
'''
print(Xone.head())
print('******************************')
'''

# 特徴量の値を格納
X = Xone.values
'''#print(Xpd.head())
#print('--------------------------')
#print(X[0:8,:])'''

#データセットをトレーニングデータセットとテストデータセットに分割
X_train, X_test, y_train, y_test, errorCheck = devisionDataset_testtrain(X, y, traintest_size)
deadErrorEnd(errorCheck)
#print(len(X_train), len(y_test))

'''print(X_train[0:5,:])
print('--------------')
print(X_test[0:5,:])
print('**************')
print('y_train',np.unique(y_train))
print('y_test ',np.unique(y_test))
print(len(y_train))
print(len(y_test))'''

#標準化のインスタンスを生成
X_train_std, X_test_std = datasetStandard(X_train, X_test)

#k分割交差検証（基本は10分割を使用）
#kfold = getKfold(y_train)
kfold = get10fold(y_train)

#print('OK')

#機械学習でGO!!!!!!
scores, times, scores10 = MLexecute(X_train, X_test, y_train, y_test,X_train_std, X_test_std)

writeResult(Xpd.columns, traintest_size*100,scores, times, scores10=scores10,trainsize=len(X_train),testsize=len(X_test))







