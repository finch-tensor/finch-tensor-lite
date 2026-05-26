from abc import ABC, abstractmethod


class Stage(ABC):

    @abstractmethod
    def validate_inputs(self, *inputs): ...

    @abstractmethod
    def transform(self, *inputs) -> tuple: ...

    def lower(self, *outputs):
        return self.ctx(*outputs)

    def __call__(self, *inputs):
        self.validate_inputs(*inputs)
        outputs = self.transform(*inputs)
        return self.lower(*outputs)
