#coding:utf-8

import glob
import csv

txtlist = glob.glob('*.txt')

flist = []
foresta = []
knna = []
svma = []
treea = []
forestt = []
knnt = []
svmt = []
treet =[]
for i in range(len(txtlist)):
    count = 0
    flist.extend([i+1])
    for readline in open('{}.txt'.format(i+1),'r'):
        #print('b')
        if readline.startswith('forest_Accuracy'):
            #print('a')
            #print(readline)
            foresta.append(float(readline[18:23]))
        if readline.startswith('knn_Accuracy'):
            # print('a')
            #print(readline)
            knna.append(float(readline[15:20]))
        if readline.startswith('svm_Accuracy'):
            #print(readline)
            svma.append(float(readline[15:20]))
        if readline.startswith('tree_Accuracy'):
            #print(readline)
            treea.append(float(readline[16:21]))
        if readline.startswith('10分割交差検証実行時間'):
            count = 1
        if not count == 0:
            if readline.startswith('\tforest:'):
                #print(readline)
                forestt.append(float(readline[8:-2]))
            if readline.startswith('\tknn   :'):
                #print(readline)
                knnt.append(float(readline[8:-2]))
            if readline.startswith('\tsvm   :'):
                #print(readline)
                svmt.append(float(readline[8:-2]))
            if readline.startswith('\ttree  :'):
                #print(readline)
                treet.append(float(readline[8:-2]))




#print(txtlist)
print(flist)
print(foresta)
print(knna)
print(svma)
print(treea)
print('\n\ntime')
print(forestt)
print(knnt)
print(svmt)
print(treet)

#with open('acu_time.csv','wb') as f:
#    csvWrite = csv.writer(f)
#    csvWrite.writerow(flist)

