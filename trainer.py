import os
import pickle

import cv2

import numpy as np

from hog import extract_features
from utils import read_labels

TRAIN_LABEL_FILE = "train/train-processed.idl"
PATCH_WIDTH = 80
PEDESTRIAN_CLASS = 1
BACKGROUND_CLASS = 0


def train(svm, label_file, model_file):
    data, classes = test_set(label_file)
    svm.fit(data, classes)
    with open(model_file, 'wb') as handle:
        pickle.dump(svm, handle, protocol=pickle.HIGHEST_PROTOCOL)


def test_set(label_file):
    labeled_data, _ = read_labels(label_file)

    labels = []
    feature_vectors = []
    for file_name, pedestrian_regions in labeled_data.items():
        image = cv2.imread(f"{os.path.dirname(label_file)}/{file_name}.png")

        # Extract features of pedestrians
        for pedestrian in pedestrian_regions:
            bottom, left, top, right = pedestrian
            window = image[bottom:top, left:right, :]
            feature_vectors.append(extract_features(window))
            labels.append(PEDESTRIAN_CLASS)

        # Extract features of background patches
        image_height = image.shape[0]
        for left, right in bg_windows(image, pedestrian_regions):
            window = image[0:image_height, left:right, :]
            features = extract_features(window)
            feature_vectors.append(features)
            labels.append(BACKGROUND_CLASS)

    return np.array(feature_vectors), np.array(labels)


def bg_windows(image, pedestrian_regions):
    image_width = image.shape[1]
    left = 0
    while left + PATCH_WIDTH <= image_width:
        right = left + PATCH_WIDTH
        overlaps_pedestrian = any(
            [p for p in pedestrian_regions if overlaps(one=(left, right), two=(p[1], p[3]))]
        )
        if not overlaps_pedestrian:
            yield left, right
        left += PATCH_WIDTH


def overlaps(one, two, threshold=40):
    one_left, one_right = one
    two_left, two_right = two
    overlap = 0
    if two_right > one_right > two_left:
        overlap = one_right - two_left
    elif two_left < one_left < two_right:
        overlap = two_right - one_left
    return overlap >= threshold
