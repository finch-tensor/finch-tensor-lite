from abc import ABC, abstractmethod


class Stage(ABC):

    @abstractmethod
    def validate_inputs(self, *inputs): ...

    @abstractmethod
    def lower(self, *inputs): ...

    def __call__(self, *inputs):
        self.validate_inputs(*inputs)
        return self.lower(*inputs)


class Form(ABC):
    """
    A Form represents the invariants that must hold for the inputs to be valid for a given stage.
    It will typically be a combination of a grammar and a set of constraints on the inputs.
    Stages should inherit from a Form to validate their inputs. 
    """
    @abstractmethod
    def validate_inputs(self, *inputs): ...

class UnvalidatedForm(Form):
    """
    UnvalidatedForm does not perform any validation on the input logic.
    This is essentially a to-do for validation and should be replaced.
    """
    def validate_inputs(self, *inputs):
        return