# Image Clustering
# Artificial Intelligence Workshop
# Pylie.com, developed by

import numpy as np


def extract_orthogonal(img):
    row_center = img[int(img.shape[0] / 2), :]
    a = len(row_center) - np.argmax(row_center < 1) - np.argmax(row_center[::-1] < 1)

    column_center = img[:, int(img.shape[1] / 2)]
    b = len(column_center) - np.argmax(column_center < 1) - np.argmax(column_center[::-1] < 1)

    return a/b


def extract_diagonal(img):
    column_center = img[:, int(img.shape[1] / 2)]
    b = len(column_center) - np.argmax(column_center < 1) - np.argmax(column_center[::-1] < 1)

    diagonal = np.diagonal(img)
    bl = np.array((np.argmax(diagonal < 1), np.argmax(diagonal < 1)))
    tr = np.array((len(diagonal) - np.argmax(diagonal[::-1] < 1), len(diagonal) - np.argmax(diagonal[::-1] < 1)))
    c = np.linalg.norm(tr - bl)

    return c/b
