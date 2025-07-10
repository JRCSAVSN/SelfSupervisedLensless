from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from pydantic import BaseModel

DATA_PATH = Path(__file__).resolve().parent.parent / 'data'


# enum for state
class BackendStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    FINISHED = "finished"
    TERMINATE = "terminate"


class ReconstructionState(BaseModel):
    current_iteration: int
    end_iteration: int
    image: Optional[bytes]
    backend_state: BackendStatus


# dataclass for config
@dataclass
class Config:
    image_size: Tuple[int, int]
    channel: int = 3
    dtype: np.dtype = np.dtype(np.float32)
    daemon: bool = True
    flip: bool = False

    def get_nbytes(self):
        nbytes = np.prod(self.image_size) * self.channel * self.dtype.itemsize
        print(nbytes)
        return int(nbytes)


@dataclass
class Parameters:
    mu1: float = 1e-6
    mu2: float = 1e-5
    mu3: float = 4e-5
    tau: float = 0.0001