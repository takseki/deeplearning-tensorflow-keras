import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

##
# X : 入力    (n_samples, n_features)
# T : 目標値  (n_samples, n_features)
# prob : 予測器の確率出力を返す関数 Y = prob(X)
#
def plot_decision_regions(X, T, prob, resolution=0.02):
    # 1-of-K を多値表現 {0,...,K-1} に直す
    if (T.shape[1] > 1):
        t = np.argmax(T, axis=1)
    else:
        t = T.flatten()     # binary
    
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(t))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    # 確率 p_k (k=0,...,K-1)
    Y = prob(np.array([xx1.ravel(), xx2.ravel()]).T)

    # 予測クラス {0,...,K-1}
    if (Y.shape[1] > 1):
        Z = np.argmax(Y, axis=1)
    else:
        Z = (Y >= 0.5)  # binary

    # 予測マップ
    plt.contourf(xx1, xx2, Z.reshape(xx1.shape), alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 入力データ
    for idx, cl in enumerate(np.unique(t)):
        plt.scatter(x=X[t == cl, 0], y=X[t == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
