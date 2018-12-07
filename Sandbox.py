# Image Clustering
# Artificial Intelligence Workshop
# Pylie.com, developed by

from scipy.ndimage import imread

import matplotlib.pyplot as plt

import os

import numpy as np

from shutil import copy2

from sklearn.cluster import KMeans

import DataSource
import FeatureLib
import PreprocLib


# Source ---
dir = os.path.dirname(os.path.realpath(__file__)) + '/Circles/'
imgs_dirs = DataSource.read_file_dirs(dir)
# ---

# Plot some images ---
img = imread(imgs_dirs[6], mode='L')
plt.imshow(img)
plt.show()

img_binary = PreprocLib.binarize(img, threshold=200)
img_square = PreprocLib.square(img_binary)
plt.imshow(img_square)
plt.show()
# ...


f1_circle_lst = []
f2_circle_lst = []

f_circle_lst = []

records = []

for img_dir in imgs_dirs:

    img = imread(img_dir, mode='L')

    # Preprocess ---
    img_binary = PreprocLib.binarize(img, threshold=200)
    img_square = PreprocLib.square(img_binary)
    # ---

    # Extract features ---
    f1 = FeatureLib.extract_orthogonal(img_square)
    f1_circle_lst.append(f1)
    f2 = FeatureLib.extract_diagonal(img_square)
    f2_circle_lst.append(f2)

    f_circle_lst.append([f1, f2])

    records.append({'dir': img_dir, 'f':[f1, f2], 'cluster': -1})
    # ---


# Squares ============================

# Source ---
dir = os.path.dirname(os.path.realpath(__file__)) + '/Squares/'
imgs_dirs = DataSource.read_file_dirs(dir)
# ---

f1_square_lst = []
f2_square_lst = []
f_square_lst = []

for img_dir in imgs_dirs:

    img = imread(img_dir, mode='L')

    # Preprocess ---
    img_binary = PreprocLib.binarize(img, threshold=200)
    img_square = PreprocLib.square(img_binary)
    # ---

    # Extract features ---
    f1 = FeatureLib.extract_orthogonal(img_square)
    f1_square_lst.append(f1)
    f2 = FeatureLib.extract_diagonal(img_square)
    f2_square_lst.append(f2)

    f_square_lst.append([f1, f2])

    records.append({'dir': img_dir, 'f': [f1, f2], 'cluster': -1})
    # ---

f_lst = f_circle_lst + f_square_lst
X = np.array(f_lst)

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

plt.scatter(np.array(f_circle_lst)[:, 0], np.array(f_circle_lst)[:, 1], s=100)
plt.scatter(np.array(f_square_lst)[:, 0], np.array(f_square_lst)[:, 1], s=80, marker='s')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='r', s=100, marker='x')
plt.xlabel('f1', size=20)
plt.ylabel('f2', size=20)
plt.show()


# Label and sink ...
cluster1_path = os.path.dirname(os.path.realpath(__file__)) + '/Clusters/Cluster1'
cluster2_path = os.path.dirname(os.path.realpath(__file__)) + '/Clusters/Cluster2'

os.makedirs(cluster1_path)
os.makedirs(cluster2_path)

for record in records:
    cluster = kmeans.predict([record['f']])
    if cluster[0] == 0:
        copy2(record['dir'], cluster1_path)
    if cluster[0] == 1:
        copy2(record['dir'], cluster2_path)
# ...
