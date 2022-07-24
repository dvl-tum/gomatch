from pathlib import Path
from typing import List, Union

import numpy as np
import torch

Device = Union[torch.device, str, None]
PathT = Union[str, Path]

TensorOrArray = Union[torch.Tensor, np.ndarray]
TensorOrArrayOrList = Union[torch.Tensor, np.ndarray, List]
