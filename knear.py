from sklearn.model_selection  import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import scipy.io as sio
import matplotlib.pyplot as plt

data = sio.loadmat('dataset.mat') 
data = data['data']
y = data[:, 0]
x = data[:, 1:]

k_range = range(0, 100)
k_error = []
# name = ['n']
# viz.line([[0.]], [1], win="data", update='replace', opts=dict(legend=name))
#循环，取k=1到k=31，查看误差效果
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x, y, cv=None, scoring='accuracy')
    Y = 1-scores.mean()
    k_error.append(Y)

# #画图，x轴为k值，y值为误差值
plt.plot(k_range, k_error)
plt.xlabel('Value of K for KNN')
plt.ylabel('Error')
plt.show()