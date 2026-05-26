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


class UnvalidatedForm(Stage):
    """
    UnvalidatedForm does not perform any validation on the input logic.
    This is essentially a to-do for validation and should be replaced.
    """
    def validate_inputs(self, *inputs):
        pass
