"""
Abstract class for all models
that consist of method that need to be implemented to model work well
"""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self):
        pass
        # For svd-based model - dim of latent vectors
        # for non-svd-based model can be None

    @abstractmethod
    def train(self, data: str, imgsz: int, epochs: int, batch: int):
        """"""
        pass

    @abstractmethod
    def val(self, data: str, imgsz: int):
        """"""
        pass

    @abstractmethod
    def predict(self, source: str):
        """"""
        pass
