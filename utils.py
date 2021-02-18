import numpy as np


def preprocess(image, constant):
    image = image[34:194, :, :]
    image = np.mean(image, axis=2, keepdims=False)
    image = image[::2, ::2]
    image = image / 256
    image = image - constant / 256
    return image
