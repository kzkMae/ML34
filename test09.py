#coding:utf-8

import itertools
a = []

for i in range(39):
    a.append(str(i))

print(a)

b = ('a', 'b')
print(type(b))

print(b)
c = 'c'
b += (c,)
print(b)

d = ()
print(type(d))

for i in range(39):
    d += (str(i+1),)

print(d)
g = []
count = 0
with open('combinationtest.txt','w') as fwrite:
    e = list(itertools.combinations(d,33))
    #print(e)
    #print(e[0], len(e[0]))
    #print(e[0][0])
    for k in e:
        f = []
        for h in k:
            f.append(h)
        fwrite.write('{}\n'.format(f))
        count += 1
        print(count)
        #print(f)
    #g.append(f)
#print(g)
#print(len(g))