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



bay = io.imread('bay.png')
boulder = io.imread('boulder.png')
great_glacier = io.imread('great_glacier.png')
salton = io.imread('salton.png')
images = {
    'bay': bay, 
    'boulder': boulder, 
    'great_glacier': great_glacier, 
    'salton': salton
}

img = np.concatenate((images["bay"], images["boulder"], images["great_glacier"], images["salton"]), 1)

rows, cols, bands = img.shape
X = img.reshape(rows*cols, bands)
classes = {
    'class0': 0, 'class1': 1, 'class2': 2}#'class3': 3,
    # 'class0': 5, 
    # 'class1': 6, 'class2': 7, 'class3': 8,
    # 'class0': 9, 'class1': 10, 'class2': 11, 'class3': 12,
    # 'class0': 13, 'class1': 14, 'class2': 15, 'class3': 16,
    # 'class0': 17, 'class1': 18, 'class2': 19, 'class3': 20,
    # 'class0': 21, 'class1': 22}
n_classes = len(classes)
# palette = np.uint8([[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 140, 51]])
palette = np.uint8([[230, 25, 75],[60, 180, 75],[255, 225, 25]])#[0, 130, 200],[245, 130, 48]])
                    # [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], 
                    # [0, 128, 128], [220, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0], 
                    # [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128], 
                    # [255, 255, 255], [0, 0, 0]])
kmeans = KMeans(n_clusters=n_classes, random_state=3).fit(X)
unsupervised = kmeans.labels_.reshape(rows, cols)
io.imshow(palette[unsupervised])
plt.show()

t0 = time.time()
supervised = n_classes*np.ones(shape=(rows, cols), dtype=np.int)
supervised[600:1024, 390:562] = classes['class1']
supervised[40:60, 40:60] = classes['class2']
supervised[100:120, 200:220] = classes['class3']
y = supervised.ravel()
train = np.flatnonzero(supervised < n_classes)
test = np.flatnonzero(supervised == n_classes)
clf = SVC(gamma='auto')
clf.fit(X[train], y[train])
t1 = time.time()
print("done training, took: " + str(t1-t0))
y[test] = clf.predict(X[test])
supervised = y.reshape(rows, cols)
io.imshow(palette[supervised])
plt.show()




# for image in images.keys():
#     img = images[image]
#     rows, cols, bands = img.shape
#     X = img.reshape(rows*cols, bands)
#     classes = {
#         'class0': 0, 'class1': 1, 'class2': 2, 'class3': 3,
#         'class0': 5, 'class1': 6, 'class2': 7, 'class3': 8,
#         'class0': 9, 'class1': 10, 'class2': 11, 'class3': 12,
#         'class0': 13, 'class1': 14, 'class2': 15, 'class3': 16,
#         'class0': 17, 'class1': 18, 'class2': 19, 'class3': 20,
#         'class0': 21, 'class1': 22}
#     n_classes = len(classes)
#     # palette = np.uint8([[0, 0, 255], [0, 255, 0], [255, 0, 0], [255, 140, 51]])
#     palette = np.uint8([[230, 25, 75],[60, 180, 75],[255, 225, 25],[0, 130, 200],[245, 130, 48], 
#                         [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], 
#                         [0, 128, 128], [220, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0], 
#                         [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128], 
#                         [255, 255, 255], [0, 0, 0]])
#     kmeans = KMeans(n_clusters=n_classes, random_state=3).fit(X)
#     unsupervised = kmeans.labels_.reshape(rows, cols)
#     images[image] = (images[image], palette[unsupervised])
#     print("done with image: " + str(image))
    

# io.imshow(images['bay'][1])
# plt.show()

# io.imshow(images['boulder'][1])
# plt.show()
# io.imshow(images['great_glacier'][1])
# plt.show()
# io.imshow(images['salton'][1])
# plt.show()



