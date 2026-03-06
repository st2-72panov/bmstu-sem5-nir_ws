from dataclasses import dataclass

import numpy as np

@dataclass()
class Pose:
    R: np.ndarray  # rotation
    T: np.ndarray  # transition
