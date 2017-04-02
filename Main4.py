#coding:utf-8

import os
import glob

txtlist = glob.glob('*.txt')

for i, file in enumerate(txtlist):
    #print(i, file)
    rename = '{}.txt'.format(i+1)
    os.rename(file,rename)

