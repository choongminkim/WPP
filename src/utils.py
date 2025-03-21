import numpy as np
import yaml


def load_config(path: str = "../../config.yaml"):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


class StainSpecficVector:
    @staticmethod
    def he():
        """_summary_
        Stain-specific values for Hematoxylin and Eosin
        """
        stain_vector = np.array(
            [
                [0.644211, 0.716556, 0.266844],  # Hematoxylin
                [0.092789, 0.954111, 0.283111],  # Eosin
            ]
        )
        return stain_vector

    @staticmethod
    def hed():
        """_summary_
        Stain-specific values for Hematoxylin, Eosin and DAB
        """
        stain_vector = np.array(
            [
                [0.650, 0.704, 0.286],  # Hematoxylin
                [0.072, 0.990, 0.105],  # Eosin
                [0.268, 0.570, 0.776],  # DAB
            ]
        )
        return stain_vector
