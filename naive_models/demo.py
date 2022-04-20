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





img = io.imread('image_part_008.jpg')
rows, cols, bands = img.shape
X = img.reshape(rows*cols, bands)
# print(X)
classes = {'water': 0, 'wetland': 1, 'farmplot': 2, 'land': 3}
n_classes = len(classes)
palette = np.uint8([[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 140, 51]])
kmeans = KMeans(n_clusters=n_classes, random_state=3).fit(X)
unsupervised = kmeans.labels_.reshape(rows, cols)
io.imshow(palette[unsupervised])
plt.show()


# t0 = time.time()
# supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
# supervised[0:500, 0:4000] = classes['water']
# supervised[2000:2200, 1300:1500] = classes['wetland']
# supervised[2750:2850, 3750:3850] = classes['farmplot']
# supervised[3750:4000, 500:1000] = classes['land']
# y = supervised.ravel()
# train = np.flatnonzero(supervised < n_classes)
# test = np.flatnonzero(supervised == n_classes)
# clf = SVC(gamma='auto')
# print(train)
# print(test)
# clf.fit(X[train], y[train])
# t1 = time.time()
# print("done training, took: " + str(t1-t0))
# y[test] = clf.predict(X[test])
# supervised = y.reshape(rows, cols)
# io.imshow(palette[supervised])
# plt.show()





