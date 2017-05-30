#coding:utf-8


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split

#headerで列名を設定（Nomeで設定の必要あり，数字の行を列名に設定）
test = pd.read_csv('test2.csv', header=0)
#test = pd.DataFrame(data='test.csv', columns=0)

#print(test.head())

#print('Vlass labels', np.unique(test['Label']))

#print(type(test))

print(test.columns)

test2 = np.unique(test['Label'])

#print(len(test2))

#for i in test2:
#    print(i)

test3 = np.unique(test['max_comand'])
#print(test3)
max_comand_mapping = {label:idx for idx, label in enumerate(test3)}

#print(max_comand_mapping)

test4 = np.unique(test['second_comand'])

#print(test4)

#tet = test3 + test4
#print(type(test3))

tet = np.r_[test3, test4]

#print(tet)

tet2 = np.unique(tet)

#print()
#print(tet2)

print(len(tet), len(tet2))

max_comand_mapping2 = {label:idx for idx, label in enumerate(tet2)}
label_mapping = {label:idx for idx, label in enumerate(np.unique(test['Label']))}
print('label mapping')
print(label_mapping)
#print(max_comand_mapping2)

test['max_comand'] = test['max_comand'].map(max_comand_mapping2)
test['Label'] = test['Label'].map(label_mapping)
#print(test.head())

test['second_comand'] = test['second_comand'].map(max_comand_mapping2)
print(test.head())

class_le = LabelEncoder()

#ohe = OneHotEncoder(categorical_features=[31])

#print(ohe.fit_transform(test))

print('-----------------------------------------------------------------------')
tes = pd.get_dummies(test)
print(tes.head())

print(tes.columns, len(tes.columns))

#t = ohe

tes2 = tes.iloc[:,[0,2]]

print(type(tes))

print(tes2.head())


X ,y = test.iloc[:, 1:2].values, test.iloc[:,-1].values
print('-------------------------')
print(y)
print('*************************')
#print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#print(y_test)
print('*************************')



#testデータ
#from sklearn.linear_model import SGDClassifier
#svm = SGDClassifier(loss='hinge')
from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=0.2, C=1.0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
print(sc.fit(X_train))
X_train_std = sc.transform(X_train)
#print(X_train_std)
#svm.fit(X_train_std, y_train)
from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40,eta0=0.1,random_state=0,shuffle=True)

#ppn.fit(X_train_std,y_train)

#k分割交差検証
from sklearn.cross_validation import StratifiedKFold

kfold = StratifiedKFold(y=y_train, n_folds=10, random_state=1)
scores = []

for k, (train, test) in enumerate(kfold):
    ppn.fit(X_train[train], y_train[train])
    score = ppn.score(X_train[test], y_train[test])
    scores.append(score)
    print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1, np.bincount(y_train[train]), score))

    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


'''

