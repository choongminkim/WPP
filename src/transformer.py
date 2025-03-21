import numpy as np
import skimage
from scipy import linalg
from typing import Optional


def rgba2rgb(image):
    image = image.astype(np.float32)
    image, alpha = image[..., :3], (image[..., 3:] / 255.0)
    image = image * alpha + (1.0 - alpha) * 255.0
    return image.astype(np.uint8)


class _OpticalDensityTransformer:
    def __init__(self, is_norm: bool = False, thres: float = 10):
        self.is_norm = is_norm
        self.thres = thres

    def _convert_transmission(self, image):
        raise NotImplementedError

    @staticmethod
    def _convert_od(tranmission):
        return -np.log10(tranmission)

    def _normalized_od(self, od):
        return od / np.log10(self.thres)

    def transform(self, image):
        transmission = self._convert_transmission(image)
        od = self._convert_od(transmission)

        if self.is_norm is True:
            od = self._normalized_od(-od)

        return od


class OpticalDensityClipTransformer(_OpticalDensityTransformer):
    def __init__(self, norm: bool = False, thres: float = 1e-6):
        super().__init__(norm, thres)

    def _convert_transmission(self, image):
        transmission = skimage.util.img_as_float(image, force_copy=True)
        return np.maximum(transmission, self.thres)


class OpticalDensityScaleTransformer(_OpticalDensityTransformer):
    def __init__(self, norm: bool = False, thres: float = 1 / 256):
        super().__init__(norm, thres)

    def _convert_transmission(self, image):
        return (image.astype(np.float64) + 1) / 256.0


class ColorDeconvolution:
    @staticmethod
    def _moore_penrose_left_inverse(stain_vector):
        return linalg.inv(stain_vector.T @ stain_vector) @ stain_vector.T

    @staticmethod
    def _moore_penrose_right_inverse(stain_vector):
        return stain_vector.T @ linalg.inv(stain_vector @ stain_vector.T)

    @staticmethod
    def _inverse(stain_vector):
        return linalg.inv(stain_vector)

    def separate(self, od, stain_vector):
        _shape = stain_vector.shape

        if _shape[0] > _shape[1]:
            _stain_vector = self._moore_penrose_left_inverse(stain_vector)

        elif _shape[0] < _shape[1]:
            _stain_vector = self._moore_penrose_right_inverse(stain_vector)

        else:
            _stain_vector = self._inverse(stain_vector)

        return od @ _stain_vector


def stain_combine(
    stains: np.array, stain_vector: np.array, is_norm: bool = False, thres: float = 10
):
    od = stains @ stain_vector

    if is_norm is False:
        od = -od

    return np.clip(np.exp(od * np.log10(thres)), a_min=0, a_max=1)
