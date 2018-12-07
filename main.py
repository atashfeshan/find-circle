# Image Clustering
# Artificial Intelligence Workshop
# Pylie.com, developed by

from scipy.ndimage import imread

import os

from shutil import copy2

from sklearn.cluster import KMeans

import DataSource
import FeatureLib
import PreprocLib


# Source ---
dir = os.path.dirname(os.path.realpath(__file__)) + '/Mix/'
imgs_dirs = DataSource.read_file_dirs(dir)
# ---

# Loop over all images ---
f_lst = []
records = []

for img_dir in imgs_dirs:

    img = imread(img_dir, mode='L')

    # Preprocess ---
    img_binary = PreprocLib.binarize(img, threshold=200)
    img_square = PreprocLib.square(img_binary)
    # ---

    # Extract features ---
    f1 = FeatureLib.extract_orthogonal(img_square)
    f2 = FeatureLib.extract_diagonal(img_square)

    f_lst.append([f1, f2])

    records.append({'dir': img_dir, 'f':[f1, f2], 'cluster': -1})
    # ---
# ---


# Train
kmeans = KMeans(n_clusters=2, random_state=0).fit(f_lst)


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