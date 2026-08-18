from .objective import Objective
import numpy as np

class BinaryLogisticObjective(Objective):

    @staticmethod
    def _sigmoid(raw_pred: np.ndarray):
        return 1 / (1 + np.exp(-raw_pred))

    @staticmethod
    def gradient(y_true: np.ndarray, raw_pred: np.ndarray):
        if len(y_true) != len(raw_pred):
            raise ValueError("y_true and raw_pred have different length.")
        p = BinaryLogisticObjective._sigmoid(raw_pred)
        return p - y_true

    @staticmethod
    def hessian(y_true: np.ndarray, raw_pred: np.ndarray):
        if len(y_true) != len(raw_pred):
            raise ValueError("y_true and raw_pred have different length.")
        p = BinaryLogisticObjective._sigmoid(raw_pred)
        return p * (1 - p)
