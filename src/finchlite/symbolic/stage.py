from abc import ABC, abstractmethod


class Stage(ABC):
    @abstractmethod
    def __call__(self, *inputs): ...
