#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np

# シグモイド関数を定義
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# 0.1間隔で-7以上7未満のデータを生成
z = np.arange(-8, 8, 0.1)
# 生成したデータでシグモイド関数を実行
phi_z = sigmoid(z)
#元のデータとシグモイド関数の出力をプロット
plt.plot(z, phi_z)

# 垂直線を追加
plt.axvline(0.0, color='k')
# y軸の上限/下限を設定
plt.ylim(-0.1,1.1)
# 軸の設定
plt.xlabel('z')
plt.ylabel('$\phi (z)$')
# y軸の目盛を追加
plt.yticks([0.0, 0.5, 1.0])
# Axesクラスのオブジェクトの取得
ax = plt.gca()
# y軸のメモリに合わせて水平グリッド線を追加
ax.yaxis.grid(True)
# グラフの表示
#plt.show()
plt.pause(5)


