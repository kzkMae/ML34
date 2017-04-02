#coding:utf-8

import os.path
import sys

#ファイルの有無を判定
def isPath(path):
    e = os.path.isfile(path)
    if not e:
        print("そんなファイルはないですよ")
        return e
    return e

# 終了用関数
def deadErrorEnd(check):
    if not check:
        print("終了します")
        sys.exit()
    return 0

# 特徴量の列がデータセット範囲内に収まっているか
def checkFeatureslist(c_size, flist):
    if flist is None:
        print('FeatureNumber No setting.')
        return True
    for i in flist:
        if i < 0:
            print("値が正しくない")
            return False
        if i > (c_size):
            print("値が正しくない")
            return False
    print('使用する特徴量の数:', len(flist))
    return True

# 使用する特徴量を表示
def printFeatures(fStringList):
    #print(fStringList)
    #print(type(fStringList))
    #print(len(fStringList))
    for i in fStringList:
        print(i,end=',\t')
    print()
    return 0




