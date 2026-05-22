from abc import ABC, abstractmethod


class Stage(ABC):

    @abstractmethod
    def validate_inputs(self, *inputs): ...

    @abstractmethod
    def transform(self, *inputs) -> tuple: ...

    @abstractmethod
    def validate_outputs(self, *outputs): ...

    def lower(self, *outputs):
        return self.ctx(*outputs)

    def __call__(self, *inputs):
        self.validate_inputs(*inputs)
        outputs = self.transform(*inputs)
        self.validate_outputs(*outputs)
        return self.lower(*outputs)


class NoTransformStage:
    """
    Mixin for stages whose ``transform`` and ``validate_outputs`` are no-ops.
    The transform forwards its inputs unchanged so that ``lower`` (or the
    downstream ``ctx``) receives the same arguments the stage was called with.
    Combine with the appropriate ``Stage`` subclass, listing ``NoTransformStage``
    first so its concrete methods win in the MRO.
    """

    def transform(self, *inputs) -> tuple:
        return inputs

    def validate_outputs(self, *outputs):
        pass
