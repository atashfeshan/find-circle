# Image Clustering
# Artificial Intelligence Workshop
# Pylie.com, developed by

import sklearn.preprocessing as pr


def binarize(img, threshold=240):
    imgb = pr.binarize(img, threshold=threshold, copy=True)
    return imgb


def square(img):
    height = img.shape[0]
    width = img.shape[1]

    if width > height:
        diff = width - height
        img_cropped = img[:, int(diff / 2): int(width - diff / 2)]
    else:
        diff = height - width
        img_cropped = img[int(diff / 2): int(height - diff / 2), :]

    return img_cropped

