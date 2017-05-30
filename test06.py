#coding:utf-8

#import
import argparse

#引数や-hのオプションを定義
parser = argparse.ArgumentParser(prog='ML用スクリプト',description='オプションと引数の説明',
                                epilog='以上')
parser.add_argument('-v','--version', action='version', version='%(prog)s version')
parser.add_argument('--A',type=int, nargs='*', help='Datasetファイルのパスを指定, 型：%(type)s，String')


# 引数を格納
arguMain = parser.parse_args()
test = arguMain.A
print(test)
print(type(test))
print(test is None)
for i in test:
    print(i)
print('a')
print(len(test))

t = 1,2,3
print(t)
print(len(t))
print(t[1])