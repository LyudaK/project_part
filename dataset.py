import os

import cv2
from sklearn import preprocessing, metrics
from sklearn.metrics.pairwise import pairwise_distances
import scipy.cluster.hierarchy as hac
from scipy.spatial.distance import squareform
import csv
import numpy as np

from facial_analysis import FacialImageProcessing

from facerec_test import is_image

imgProcessing = FacialImageProcessing()

subject_ids = []
face_features = []
ages = []
genders = []
all_facial_images = []


def rect_intersection_square(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[2], b[2]) - x
    h = min(a[3], b[3]) - y
    if w < 0 or h < 0:
        w = h = 0
    return w * h


with open(os.path.join('GallagherDatasetGT.txt'), 'r') as csvfile:
    directory='D://faces'
    dirs_and_files = np.array([[dir, os.path.join(dir, file)] for dir in next(os.walk(directory))[1] for file in next(os.walk(os.path.join(directory, dir)))[2] if is_image(file)])
    dirs = dirs_and_files[:, 0]
    files = dirs_and_files[:, 1]
    print('dirs', dirs)
    print('files', files)
    idx=0
    for f in files:
        print(f)

        face_img = cv2.imread(os.path.join(directory,f))
        #cv2.imshow('',face_img)
        #cv2.waitKey(0)
        age, gender, features = imgProcessing.age_gender_fun(face_img)

        all_facial_images.append(face_img)
        print('id',dirs[idx])
        subject_ids.append(dirs[idx])
        face_features.append(features)
        ages.append(age)
        genders.append(gender)
        idx=idx+1

    print(face_features)
    print('subject_ids',subject_ids)


# B-cubed
def fscore(p_val, r_val, beta=1.0):
    """Computes the F_{beta}-score of given precision and recall values."""
    return (1.0 + beta ** 2) * (p_val * r_val / (beta ** 2 * p_val + r_val))


def mult_precision(el1, el2, cdict, ldict):
    """Computes the multiplicity precision for two elements."""
    return min(len(cdict[el1] & cdict[el2]), len(ldict[el1] & ldict[el2])) \
           / float(len(cdict[el1] & cdict[el2]))


def mult_recall(el1, el2, cdict, ldict):
    """Computes the multiplicity recall for two elements."""
    return min(len(cdict[el1] & cdict[el2]), len(ldict[el1] & ldict[el2])) \
           / float(len(ldict[el1] & ldict[el2]))


def precision(cdict, ldict):
    """Computes overall extended BCubed precision for the C and L dicts."""
    return np.mean([np.mean([mult_precision(el1, el2, cdict, ldict) \
                             for el2 in cdict if cdict[el1] & cdict[el2]]) for el1 in cdict])


def recall(cdict, ldict):
    """Computes overall extended BCubed recall for the C and L dicts."""
    return np.mean([np.mean([mult_recall(el1, el2, cdict, ldict) \
                             for el2 in cdict if ldict[el1] & ldict[el2]]) for el1 in cdict])


def get_BCubed_set(y_vals):
    dic = {}
    for i, y in enumerate(y_vals):
        dic[i] = set([y])
    return dic


def BCubed_stat(y_true, y_pred, beta=1.0):
    cdict = get_BCubed_set(y_true)
    ldict = get_BCubed_set(y_pred)
    p = precision(cdict, ldict)
    r = recall(cdict, ldict)
    f = fscore(p, r, beta)
    return (p, r, f)


label_enc = preprocessing.LabelEncoder()
label_enc.fit(subject_ids)
y_true = label_enc.transform(subject_ids)

face_features = np.array(face_features)
ages = np.array(ages)
genders = np.array(genders)
X_norm = preprocessing.normalize(face_features, norm='l2')
dist_matrix = pairwise_distances(X_norm)

distanceThreshold = 0.8 #0.97
clusteringMethod = 'average'
condensed_dist_matrix = squareform(dist_matrix, checks=False)
z = hac.linkage(condensed_dist_matrix, method=clusteringMethod)
y_pred = hac.fcluster(z, distanceThreshold, 'distance')

homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(y_true, y_pred)
# fm=metrics.fowlkes_mallows_score(y_true, y_pred)
bcubed_precision, bcubed_recall, bcubed_fmeasure = BCubed_stat(y_true, y_pred)
print('Galagher dataset: # classes=%d #clusters=%d' % (len(np.unique(y_true)), len(np.unique(y_pred))))
print('homogeneity=%0.3f, completeness=%0.3f' % (homogeneity, completeness))
print('BCubed precision=%0.3f, recall=%0.3f' % (bcubed_precision, bcubed_recall))

imgProcessing.close()
