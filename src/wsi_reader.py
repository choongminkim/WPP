from pathlib import Path
from typing import Tuple

from openslide import OpenSlide
import numpy as np

from src.utils import rgba2rgb


# def read_wsi(path: str, level: int = 0):
#     try:
#         wsi = OpenSlide(path)
#         image = wsi.read_region((0, 0), level, wsi.level_dimensions[level])
#         image = np.array(image)
#         return image
#     except Exception as e:
#         print(e)


class WSIReader(object):
    def __init__(self, path: str):
        self.slide = OpenSlide(filename=path)

        self.level_d = self.slide.level_dimensions

    def _get_best_level_for_downsample(self, level: int):
        return self.slide.get_best_level_for_downsample(level)

    def read_array(
        self,
        location: Tuple[int, int] = (0, 0),
        level: int = 1,
        size: Tuple[int, int] = None,
    ):

        level = self._get_best_level_for_downsample(level)

        if size is None:
            size = self.level_d[level]

        rgb = np.array(
            self.slide.read_region(location=location, level=level, size=size)
        )

        if rgb.shape[-1] == 4:
            rgb = rgba2rgb(rgb)

        return rgb
