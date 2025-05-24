from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from ..symbolic import Term


@dataclass(eq=True, frozen=True)
class AssemblyNode(Term):
    """
    AssemblyNode

    Represents a FinchAssembly IR node. FinchAssembly is the final intermediate
    representation before code generation (translation to the output language).
    It is a low-level imperative description of the program, with control flow,
    linear memory regions called "buffers", and explicit memory management.
    """

    @staticmethod
    @abstractmethod
    def is_expr():
        """Determines if the node is expresion."""
        ...

    @staticmethod
    @abstractmethod
    def is_stateful():
        """Determines if the node is stateful."""
        ...

    @classmethod
    def head(cls):
        """Returns the head of the node."""
        return cls

    def children(self):
        """Returns the children of the node."""
        raise Exception(f"`children` isn't supported for {self.__class__}.")

    @classmethod
    def make_term(cls, head, *args):
        """Creates a term with the given head and arguments."""
        return head(*args)


@dataclass(eq=True, frozen=True)
class Immediate(AssemblyNode):
    """
    Represents the literal value `val`.

    Attributes:
        val: The literal value.
    """

    val: Any

    @staticmethod
    def is_expr():
        """Determines if the node is an expression."""
        return False

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False


@dataclass(eq=True, frozen=True)
class Variable(AssemblyNode):
    """
    Represents a logical AST expression for a variable named `name`.

    Attributes:
        name: The name of the variable.
    """

    name: str

    @staticmethod
    def is_expr():
        """Determines if the node is an expression."""
        return False

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False

    def children(self):
        """Returns the children of the node."""
        return [self.name]


@dataclass(eq=True, frozen=True)
class Symbolic(AssemblyNode):
    """
    Represents a logical AST expression for a symbolic object `obj`.

    Attributes:
        obj: The tensor object.
    """

    obj: AssemblyNode

    @staticmethod
    def is_expr():
        """Determines if the node is an expression."""
        return True

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False

    def children(self):
        """Returns the children of the node."""
        return [self.obj]


@dataclass(eq=True, frozen=True)
class Load(AssemblyNode):
    """
    Represents loading a value from a buffer at a given index.

    Attributes:
        buffer: The buffer to load from.
        index: The index to load at.
    """

    buffer: AssemblyNode
    index: AssemblyNode

    @staticmethod
    def is_expr():
        return True

    @staticmethod
    def is_stateful():
        return False

    def children(self):
        return [self.buffer, self.index]


@dataclass(eq=True, frozen=True)
class Store(AssemblyNode):
    """
    Represents storing a value into a buffer at a given index.

    Attributes:
        buffer: The buffer to store into.
        index: The index to store at.
        value: The value to store.
    """

    buffer: AssemblyNode
    index: AssemblyNode
    value: AssemblyNode

    @staticmethod
    def is_expr():
        return True

    @staticmethod
    def is_stateful():
        return True

    def children(self):
        return [self.buffer, self.index, self.value]


@dataclass(eq=True, frozen=True)
class Resize(AssemblyNode):
    """
    Represents resizing a buffer to a new size.

    Attributes:
        buffer: The buffer to resize.
        new_size: The new size for the buffer.
    """

    buffer: AssemblyNode
    new_size: AssemblyNode

    @staticmethod
    def is_expr():
        return True

    @staticmethod
    def is_stateful():
        return True

    def children(self):
        return [self.buffer, self.new_size]


@dataclass(eq=True, frozen=True)
class Length(AssemblyNode):
    """
    Represents getting the length of a buffer.

    Attributes:
        buffer: The buffer whose length is queried.
    """

    buffer: AssemblyNode

    @staticmethod
    def is_expr():
        return True

    @staticmethod
    def is_stateful():
        return False

    def children(self):
        return [self.buffer]


@dataclass(eq=True, frozen=True)
class Call(AssemblyNode):
    """
    Represents an expression for calling the function `op` on `args...`.

    Attributes:
        op: The function to call.
        args: The arguments to call on the function.
    """

    op: AssemblyNode
    args: tuple[AssemblyNode, ...]

    @staticmethod
    def is_expr():
        """Determines if the node is an expression."""
        return True

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False

    def children(self):
        """Returns the children of the node."""
        return [self.op, *self.args]

    @classmethod
    def make_term(cls, head, op, *args):
        return head(op, args)


@dataclass(eq=True, frozen=True)
class Function(AssemblyNode):
    """
    Represents a logical AST statement that defines a function `fun` on the
    arguments `args...`.

    Attributes:
        name: The name of the function to define.
        args: The arguments to the function.
        body: The body of the function. If it does not contain a return statement,
            the function returns the value of `body`.
    """

    name: str
    args: tuple[AssemblyNode, ...]
    body: AssemblyNode

    @staticmethod
    def is_expr():
        """Determines if the node is an expression."""
        return True

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return False

    def children(self):
        """Returns the children of the node."""
        return [self.name, *self.args, self.body]

    def make_term(self, head, *args):
        """Creates a term with the given head and arguments."""
        return head(*args[1], args[2:-2], args[-1])


@dataclass(eq=True, frozen=True)
class ForLoop(AssemblyNode):
    """
    Represents a for loop that iterates over a range of values.

    Attributes:
        var: The loop variable.
        start: The starting value of the range.
        end: The ending value of the range.
        body: The body of the loop to execute.
    """

    var: AssemblyNode
    start: AssemblyNode
    end: AssemblyNode
    body: AssemblyNode

    @staticmethod
    def is_expr():
        """Determines if the node is an expression."""
        return False

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return True

    def children(self):
        """Returns the children of the node."""
        return [self.var, self.start, self.end, self.body]

    @classmethod
    def make_term(cls, head, *args):
        return head(*args)


@dataclass(eq=True, frozen=True)
class BufferLoop(AssemblyNode):
    """
    Represents a loop that iterates over the elements of a buffer.

    Attributes:
        buffer: The buffer to iterate over.
        var: The loop variable for each element in the buffer.
        body: The body of the loop to execute for each element.
    """

    buffer: AssemblyNode
    var: AssemblyNode
    body: AssemblyNode

    @staticmethod
    def is_expr():
        """Determines if the node is an expression."""
        return False

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return True

    def children(self):
        """Returns the children of the node."""
        return [self.buffer, self.var, self.body]

    @classmethod
    def make_term(cls, head, *args):
        return head(*args)


@dataclass(eq=True, frozen=True)
class WhileLoop(AssemblyNode):
    """
    Represents a while loop that executes as long as the condition is true.

    Attributes:
        condition: The condition to evaluate for the loop to continue.
        body: The body of the loop to execute.
    """

    condition: AssemblyNode
    body: AssemblyNode

    @staticmethod
    def is_expr():
        """Determines if the node is an expression."""
        return False

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return True

    def children(self):
        """Returns the children of the node."""
        return [self.condition, self.body]

    @classmethod
    def make_term(cls, head, *args):
        return head(*args)


@dataclass(eq=True, frozen=True)
class Assign(AssemblyNode):
    """
    Represents a logical AST statement that evaluates `rhs`, binding the result
    to `lhs`.

    Attributes:
        type: The type of the new binding.
        lhs: The left-hand side of the binding.
        rhs: The right-hand side to evaluate.
    """

    type: AssemblyNode
    lhs: AssemblyNode
    rhs: AssemblyNode

    @staticmethod
    def is_expr():
        """Determines if the node is an expression."""
        return True

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return True

    def children(self):
        """Returns the children of the node."""
        return [self.type, self.lhs, self.rhs]


@dataclass(eq=True, frozen=True)
class Return(AssemblyNode):
    """
    Represents a return statement that returns `arg` from the current function.
    Halts execution of the function body.

    Attributes:
        arg: The argument to return.
    """

    args: tuple[AssemblyNode, ...]

    @staticmethod
    def is_expr():
        """Determines if the node is an expression."""
        return True

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return True

    def children(self):
        """Returns the children of the node."""
        return [self.arg]

    @classmethod
    def make_term(cls, head, *args):
        return head(args)


@dataclass(eq=True, frozen=True)
class Block(AssemblyNode):
    """
    Represents a statement that executes a sequence of statements `bodies...`.

    Attributes:
        bodies: The sequence of statements to execute.
    """

    bodies: tuple[AssemblyNode, ...] = ()

    @staticmethod
    def is_expr():
        """Determines if the node is an expression."""
        return True

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return True

    def children(self):
        """Returns the children of the node."""
        return [*self.bodies]

    @classmethod
    def make_term(cls, head, *val):
        return head(val)


@dataclass(eq=True, frozen=True)
class Module(AssemblyNode):
    """
    Represents a group of functions. This is the toplevel translation unit for
    FinchAssembly.

    Attributes:
        funcs: The functions defined in the module.
        main: The main function of the module.
    """

    funcs: tuple[AssemblyNode, ...] = ()
    main: AssemblyNode = None

    @staticmethod
    def is_expr():
        """Determines if the node is an expression."""
        return True

    @staticmethod
    def is_stateful():
        """Determines if the node is stateful."""
        return True

    def children(self):
        """Returns the children of the node."""
        return [*self.funcs, self.main]

    @classmethod
    def make_term(cls, head, *val):
        return head(val[1:-2], val[-1])
