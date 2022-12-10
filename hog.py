from skimage.feature import hog


def extract_features(image):
    """

    :param image: is ndarray, for example: np.array(PIL.Image.open(file))
    :return: ndarray with HOG
    """
    features = hog(image, orientations=8, pixels_per_cell=(16, 16),
                   cells_per_block=(1, 1), visualize=False, channel_axis=-1)
    return features
