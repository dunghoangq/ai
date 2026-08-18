from abc import ABC, abstractmethod
import numpy as np

class Objective(ABC):

    @staticmethod
    @abstractmethod
    def gradient(y_true: np.ndarray, raw_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Objective class requires gradient()")

    @staticmethod
    @abstractmethod
    def hessian(y_true: np.ndarray, raw_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Objective class requires hessian()")