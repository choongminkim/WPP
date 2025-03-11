import numpy as np
import yaml


def rgba2rgb(image):
    image = image.astype(np.float32)
    image, alpha = image[..., :3], (image[..., 3:] / 255.0)
    image = image * alpha + (1.0 - alpha) * 255.0
    return image.astype(np.uint8)


def load_config(path: str = "../../config.yaml"):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


class StainConcentrationMatrix:
    @staticmethod
    def he():
        """_summary_
        Stain-specific values for Hematoxylin and Eosin
        """
        cmatrix = np.array(
            [
                [0.644211, 0.716556, 0.266844],  # Hematoxylin
                [0.092789, 0.954111, 0.283111],  # Eosin
                [0.0, 0.0, 0.0],  # None
            ]
        )
        return cmatrix

    @staticmethod
    def hed():
        """_summary_
        Stain-specific values for Hematoxylin, Eosin and DAB
        """
        cmatrix = np.array(
            [
                [0.650, 0.704, 0.286],  # Hematoxylin
                [0.072, 0.990, 0.105],  # Eosin
                [0.268, 0.570, 0.776],  # DAB
            ]
        )
        return cmatrix
