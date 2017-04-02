#coding:utf-8

import subprocess
import itertools

choiceNum = 10

mlcomand = ['C:\Python34\python.exe', 'C:/Users/Max/Desktop/M1/８．GitHub/ML34/Main.py', '../DataMain.csv' ,'0.3', '--Features']

dellist = [0,1,2,3,6,7,10,11,12,13,14,15,16,17,18,19,21,22,23,24,25,27,30,32,34,35,36,37,38]

featureslist = ()

for i in range(39):
    if i in dellist:
        continue
    featureslist += (str(i),)

#print(featureslist)
comblist = list(itertools.combinations(featureslist,choiceNum))

print(len(comblist))

for features in comblist:
    mlcomandfeatues = mlcomand + list(features)
    #print(list(features))
    print(mlcomandfeatues)
    subprocess.call(mlcomandfeatues)





