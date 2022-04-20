from io import UnsupportedOperation
import numpy as np
from skimage import io
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import ssl
import matplotlib
from matplotlib import pyplot as plt
import time
matplotlib.use('TkAgg')
io.use_plugin('matplotlib')
ssl._create_default_https_context = ssl._create_unverified_context


img = io.imread('https://i.stack.imgur.com/TFOv7.png')
rows, cols, bands = img.shape
bands = 3
classes = {'water': 0, 'wetland': 1, 'farmplot': 2}
n_classes = len(classes)
palette = np.uint8([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
X = img.reshape(rows*cols, bands)
kmeans = KMeans(n_clusters=n_classes, random_state=3).fit(X)
unsupervised = kmeans.labels_.reshape(rows, cols)
io.imshow(palette[unsupervised])
plt.show()

t0 = time.time()
supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
supervised[200:220, 150:170] = classes['water']
supervised[40:60, 40:60] = classes['wetland']
supervised[100:120, 200:220] = classes['farmplot']
y = supervised.ravel()
train = np.flatnonzero(supervised < n_classes)
test = np.flatnonzero(supervised == n_classes)
print(train)
print(test)
clf = SVC(gamma='auto')
clf.fit(X[train], y[train])
t1 = time.time()
print("done training, took: " + str(t1-t0))
y[test] = clf.predict(X[test])
supervised = y.reshape(rows, cols)
io.imshow(palette[supervised])
plt.show()


