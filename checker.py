import os
import pickle

import cv2
import numpy as np

from hog import extract_features
from trainer import PATCH_WIDTH, PEDESTRIAN_CLASS
from utils import read_labels

WINDOW_WIDTH = 20


def find_pedestrians(model_file, image_path):
    image = cv2.imread(image_path)
    window_features = []
    window_corners = []
    height = image.shape[0]
    for left, right in generate_windows(image, PATCH_WIDTH):
        image_slice = image[0:height, left:right, :]
        window_features.append(extract_features(image_slice))
        window_corners.append((0, left, height, right))
    window_features = np.array(window_features)
    result = load_svm(model_file).predict(window_features)
    regions = find_positive_windows(result, window_corners)
    return render_bounds(image, regions)


def check(model_file, label_file):
    labels, expected_pedestrians = read_labels(label_file)

    true_positives = 0
    false_positives = 0

    for file_name, pedestrian_regions in labels.items():
        image = cv2.imread(f"{os.path.dirname(label_file)}/{file_name}.png")

        window_features = []
        window_corners = []

        height = image.shape[0]
        for left, right in generate_windows(image, PATCH_WIDTH):
            image_slice = image[0:height, left:right, :]
            window_features.append(extract_features(image_slice))
            window_corners.append((0, left, height, right))
        window_features = np.array(window_features)

        result = load_svm(model_file).predict(window_features)

        tp, fp, tp_findings = test_image(result, pedestrian_regions, window_corners)

        true_positives += tp
        false_positives += fp

    recall = true_positives / expected_pedestrians
    precision = true_positives / (true_positives + false_positives)
    print(f"Recall: {true_positives / expected_pedestrians}")
    print(f"Precision: {precision}")
    return recall, precision, false_positives


def render_bounds(image, findings):
    if len(findings) == 0:
        return
    temp_image = np.copy(image)
    for region in findings:
        cv2.rectangle(img=temp_image,
                      pt1=(region[0], region[1]),
                      pt2=(region[2], region[3]),
                      color=(0, 255, 0),
                      thickness=2)
    return temp_image


def find_positive_windows(svm_output, window_corners):
    positive_windows = []
    for i, clazz in enumerate(svm_output):
        if clazz != PEDESTRIAN_CLASS:
            continue
        found_window = window_corners[i]
        positive_windows.append((found_window[1], found_window[0],
                                 found_window[3], found_window[2]))
    return positive_windows


def test_image(classification, pedestrian_regions, window_corners):
    positive_windows = {}
    found_pedestrians = set()
    false_positives = 0
    for i, clazz in enumerate(classification):
        if clazz != PEDESTRIAN_CLASS:
            continue
        found_window = window_corners[i]
        for pedestrian in pedestrian_regions:
            if abs(found_window[1] - pedestrian[1]) < PATCH_WIDTH / 2:
                found_pedestrians.add(pedestrian)
                if pedestrian not in positive_windows:
                    positive_windows[pedestrian] = []
                positive_windows[pedestrian].append((found_window[1], found_window[0],
                                                     found_window[3], found_window[2]))
        if len(found_pedestrians) == 0:
            false_positives += 1

    if false_positives == 0 and len(found_pedestrians) == 0:
        pass
    return len(found_pedestrians), false_positives, positive_windows


def load_svm(coefficients_file):
    with open(coefficients_file, 'rb') as handle:
        svm = pickle.load(handle)
    return svm


def generate_windows(image, step):
    width = image.shape[1]
    left = 0
    while left + step <= width:
        yield left, left + step
        left += WINDOW_WIDTH
