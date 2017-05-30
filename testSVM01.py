#coding:utf-8

'''
非線形分離可能なケースの場合のSVM（サポートベクターマシン）
'''

#import
from sklearn.svm import SVC
import time
import numpy as np
from sklearn.cross_validation import StratifiedKFold



print('データセットを入力')

print('特徴量を抽出')

print('クラスラベルの取得')

#kfold = StratifiedKFold(y=y_train,n_folds=10, random_state=1)

scores = []

for k, (train, test) in enumerate(kfold):


start = time.time()



#線形SVMのインスタンスを生成
svm = SVC(kernel='linear',C=1.0, random_state=0)

time.sleep(1)
end = time.time()

#print('計測時間：%lf' %(end - start))
#print(start)

#print(end)

