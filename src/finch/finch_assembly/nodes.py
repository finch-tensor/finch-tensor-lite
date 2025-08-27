from abc import abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

from ..algebra import return_type
from ..symbolic import Context, Term, TermTree, literal_repr
from ..util import qual_str
from .buffer import element_type, length_type


class AssemblyNode(Term):
    """
    AssemblyNode

    Represents a FinchAssembly IR node. FinchAssembly is the final intermediate
    representation before code generation (translation to the output language).
    It is a low-level imperative description of the program, with control flow,
    linear memory regions called "buffers", and explicit memory management.
    """

    @classmethod
    def head(cls):
        """Returns the head of the node."""
        return cls

    @classmethod
    def make_term(cls, head, *args):
        """Creates a term with the given head and arguments."""
        return head.from_children(*args)

    @classmethod
    def from_children(cls, *children):
        """
        Creates a term from the given children. This is used to create terms
        from the children of a node.
        """
        return cls(*children)

    def __str__(self):
        """Returns a string representation of the node."""
        ctx = AssemblyPrinterContext()
        ctx(self)
        return ctx.emit()


class AssemblyTree(AssemblyNode, TermTree):
    @property
    def children(self):
        """Returns the children of the node."""
        raise Exception(f"`children` isn't supported for {self.__class__}.")


class AssemblyExpression(AssemblyNode):
    @property
    @abstractmethod
    def result_format(self):
        """Returns the type of the expression."""
        ...


@dataclass(eq=True, frozen=True)
class Literal(AssemblyExpression):
    """
    Represents the literal value `val`.

    Attributes:
        val: The literal value.
    """

    val: Any

    @property
    def result_format(self):
        """Returns the type of the expression."""
        return type(self.val)

    def __repr__(self) -> str:
        return literal_repr(type(self).__name__, asdict(self))


@dataclass(eq=True, frozen=True)
class Variable(AssemblyExpression):
    """
    Represents a logical AST expression for a variable named `name`, which
    will hold a value of type `type`.

    Attributes:
        name: The name of the variable.
        type: The type of the variable.
    """

    name: str
    type: Any

    @property
    def result_format(self):
        """Returns the type of the expression."""
        return self.type

    def __repr__(self) -> str:
        return literal_repr(type(self).__name__, asdict(self))


@dataclass(eq=True, frozen=True)
class Stack(AssemblyExpression):
    """
    A logical AST expression representing an object using a set `obj` of
    expressions, variables, and literals in the target language.

    Attributes:
        obj: The object referencing symbolic variables defined in the target language.
        type: The type of the symbolic object.
    """

    obj: Any
    type: Any

    @property
    def result_format(self):
        """Returns the type of the expression."""
        return self.type


@dataclass(eq=True, frozen=True)
class Slot(AssemblyExpression):
    """
    Represents a register to a symbolic object. Using a register in an
    expression creates a copy of the object.

    Attributes:
        name: The name of the symbolic object to register.
        type: The type of the symbolic object.
    """

    name: str
    type: Any

    @property
    def result_format(self):
        """Returns the type of the expression."""
        return self.type

    def __repr__(self) -> str:
        return literal_repr(type(self).__name__, asdict(self))


@dataclass(eq=True, frozen=True)
class Unpack(AssemblyTree):
    """
    Attempts to convert `rhs` into a symbolic, which can be registerd with
    `lhs`. The original object must not be accessed or modified until the
    corresponding `Repack` node is reached.

    Attributes:
        lhs: The symbolic object to write to.
        rhs: The original object to read from.
    """

    lhs: Slot
    rhs: AssemblyExpression

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.lhs, self.rhs]


@dataclass(eq=True, frozen=True)
class Repack(AssemblyTree):
    """
    Registers updates from a symbolic object `val` with the original
    object. The original object may now be accessed and modified.

    Attributes:
        slot: The symbolic object to read from.
    """

    val: Slot

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.val]


@dataclass(eq=True, frozen=True)
class Assign(AssemblyTree):
    """
    Represents a logical AST statement that evaluates `rhs`, binding the result
    to `lhs`.

    Attributes:
        lhs: The left-hand side of the binding.
        rhs: The right-hand side to evaluate.
    """

    lhs: Variable | Stack
    rhs: AssemblyExpression

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.lhs, self.rhs]


@dataclass(eq=True, frozen=True)
class GetAttr(AssemblyExpression, AssemblyTree):
    """
    Represents a getter for an attribute `attr` of an object `obj`.
    Attributes:
        obj: The object to get the attribute from.
        attr: The name of the attribute to get.
    """

    obj: AssemblyExpression
    attr: Literal

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.obj, self.attr]

    @property
    def result_format(self):
        """Returns the type of the expression."""
        return dict(self.obj.result_format.struct_fields)[self.attr.val]


@dataclass(eq=True, frozen=True)
class SetAttr(AssemblyTree):
    """
    Represents a setter for an attribute `attr` of an object `obj`.
    Attributes:
        obj: The object to set the attribute on.
        attr: The name of the attribute to set.
        value: The value to set the attribute to.
    """

    obj: AssemblyExpression
    attr: Literal
    value: AssemblyExpression

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.obj, self.attr, self.value]


@dataclass(eq=True, frozen=True)
class Call(AssemblyExpression, AssemblyTree):
    """
    Represents an expression for calling the function `op` on `args...`.

    Attributes:
        op: The function to call.
        args: The arguments to call on the function.
    """

    op: Literal
    args: tuple[AssemblyNode, ...]

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.op, *self.args]

    @classmethod
    def from_children(cls, op, *args):
        return cls(op, args)

    @property
    def result_format(self):
        """Returns the type of the expression."""
        arg_types = [arg.result_format for arg in self.args]
        return return_type(self.op.val, *arg_types)


@dataclass(eq=True, frozen=True)
class Load(AssemblyExpression, AssemblyTree):
    """
    Represents loading a value from a buffer at a given index.

    Attributes:
        buffer: The buffer to load from.
        index: The index to load at.
    """

    buffer: AssemblyExpression
    index: AssemblyExpression

    @property
    def children(self):
        return [self.buffer, self.index]

    @property
    def result_format(self):
        """Returns the type of the expression."""
        return element_type(self.buffer.result_format)


@dataclass(eq=True, frozen=True)
class Store(AssemblyTree):
    """
    Represents storing a value into a buffer at a given index.

    Attributes:
        buffer: The buffer to store into.
        index: The index to store at.
        value: The value to store.
    """

    buffer: AssemblyExpression
    index: AssemblyExpression
    value: AssemblyExpression

    @property
    def children(self):
        return [self.buffer, self.index, self.value]


@dataclass(eq=True, frozen=True)
class Resize(AssemblyTree):
    """
    Represents resizing a buffer to a new size.

    Attributes:
        buffer: The buffer to resize.
        new_size: The new size for the buffer.
    """

    buffer: AssemblyExpression
    new_size: AssemblyExpression

    @property
    def children(self):
        return [self.buffer, self.new_size]


@dataclass(eq=True, frozen=True)
class Length(AssemblyExpression, AssemblyTree):
    """
    Represents getting the length of a buffer.

    Attributes:
        buffer: The buffer whose length is queried.
    """

    buffer: AssemblyExpression

    @property
    def children(self):
        return [self.buffer]

    @property
    def result_format(self):
        """Returns the type of the expression."""
        return length_type(self.buffer.result_format)


@dataclass(eq=True, frozen=True)
class ForLoop(AssemblyTree):
    """
    Represents a for loop that iterates over a range of values.

    Attributes:
        var: The loop variable.
        start: The starting value of the range.
        end: The ending value of the range.
        body: The body of the loop to execute.
    """

    var: Variable
    start: AssemblyExpression
    end: AssemblyExpression
    body: AssemblyNode

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.var, self.start, self.end, self.body]


@dataclass(eq=True, frozen=True)
class BufferLoop(AssemblyTree):
    """
    Represents a loop that iterates over the elements of a buffer.

    Attributes:
        buffer: The buffer to iterate over.
        var: The loop variable for each element in the buffer.
        body: The body of the loop to execute for each element.
    """

    buffer: AssemblyExpression
    var: Variable
    body: AssemblyNode

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.buffer, self.var, self.body]


@dataclass(eq=True, frozen=True)
class WhileLoop(AssemblyTree):
    """
    Represents a while loop that executes as long as the condition is true.

    Attributes:
        condition: The condition to evaluate for the loop to continue.
        body: The body of the loop to execute.
    """

    condition: AssemblyExpression
    body: AssemblyNode

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.condition, self.body]


@dataclass(eq=True, frozen=True)
class If(AssemblyTree):
    """
    Represents an if statement that executes the body if the condition is true.

    Attributes:
        condition: The condition to evaluate for the if to execute the body.
        body: The body of the if statement to execute.
    """

    condition: AssemblyExpression
    body: AssemblyNode

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.condition, self.body]


@dataclass(eq=True, frozen=True)
class IfElse(AssemblyTree):
    """
    Represents an if-else statement that executes the body if the condition
    is true, otherwise executes else_body.

    Attributes:
        condition: The condition to evaluate for the if to execute the body.
        body: The body of the if statement to execute.
        else_body: An alternative body to execute if the condition is false.
    """

    condition: AssemblyExpression
    body: AssemblyNode
    else_body: AssemblyNode

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.condition, self.body, self.else_body]


@dataclass(eq=True, frozen=True)
class Function(AssemblyTree):
    """
    Represents a logical AST statement that defines a function `fun` on the
    arguments `args...`.

    Attributes:
        name: The name of the function to define as a variable typed with the
            return type of this function.
        args: The arguments to the function.
        body: The body of the function. If it does not contain a return statement,
            the function returns the value of `body`.
    """

    name: Variable
    args: tuple[Variable, ...]
    body: AssemblyNode

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.name, *self.args, self.body]

    @classmethod
    def from_children(cls, name, *args, body):
        """Creates a term with the given head and arguments."""
        return cls(name, args, body)


@dataclass(eq=True, frozen=True)
class Return(AssemblyTree):
    """
    Represents a return statement that returns `arg` from the current function.
    Halts execution of the function body.

    Attributes:
        arg: The argument to return.
    """

    arg: AssemblyExpression

    @property
    def children(self):
        """Returns the children of the node."""
        return [self.arg]


@dataclass(eq=True, frozen=True)
class Break(AssemblyTree):
    """
    Represents a break statement that exits the current loop.
    """

    @property
    def children(self):
        """Returns the children of the node."""
        return []


@dataclass(eq=True, frozen=True)
class Block(AssemblyTree):
    """
    Represents a statement that executes a sequence of statements `bodies...`.

    Attributes:
        bodies: The sequence of statements to execute.
    """

    bodies: tuple[AssemblyNode, ...] = ()

    @property
    def children(self):
        """Returns the children of the node."""
        return [*self.bodies]

    @classmethod
    def from_children(cls, *bodies):
        return cls(bodies)


@dataclass(eq=True, frozen=True)
class Module(AssemblyTree):
    """
    Represents a group of functions. This is the toplevel translation unit for
    FinchAssembly.

    Attributes:
        funcs: The functions defined in the module.
    """

    funcs: tuple[AssemblyNode, ...]

    @property
    def children(self):
        """Returns the children of the node."""
        return [*self.funcs]

    @classmethod
    def from_children(cls, *funcs):
        return cls(funcs)


@dataclass(eq=True, frozen=True)
class Print(AssemblyTree):
    """
    Print a message along with an expression.

    Attributes:
        message: The message to be output.
        args: The expression to be printed.
    """

    message: Variable
    args: Variable

    @property
    def children(self):
        """Returns the children of the node."""
        return []


class AssemblyPrinterContext(Context):
    def __init__(self, tab="    ", indent=0):
        super().__init__()
        self.tab = tab
        self.indent = indent

    @property
    def feed(self) -> str:
        return self.tab * self.indent

    def emit(self):
        return "\n".join([*self.preamble, *self.epilogue])

    def block(self) -> "AssemblyPrinterContext":
        blk = super().block()
        blk.indent = self.indent
        blk.tab = self.tab
        return blk

    def subblock(self):
        blk = self.block()
        blk.indent = self.indent + 1
        return blk

    def __call__(self, prgm: AssemblyNode):
        feed = self.feed
        match prgm:
            case Literal(value):
                return qual_str(value)
            case Variable(name, _):
                return str(name)
            case Assign(Variable(var_n, var_t), val):
                self.exec(f"{feed}{var_n}: {qual_str(var_t)} = {self(val)}")
                return None
            case GetAttr(obj, attr):
                return f"getattr({obj}, {attr})"
            case SetAttr(obj, attr, val):
                return f"setattr({obj}, {attr})"
            case Call(Literal(_) as lit, args):
                return f"{self(lit)}({', '.join(self(arg) for arg in args)})"
            case Unpack(Slot(var_n, var_t), val):
                self.exec(f"{feed}{var_n}: {qual_str(var_t)} = unpack({self(val)})")
                return None
            case Repack(Slot(var_n, var_t)):
                self.exec(f"{feed}repack({var_n})")
                return None
            case Load(buf, idx):
                return f"load({self(buf)}, {self(idx)})"
            case Slot(name, type_):
                return f"slot({name}, {qual_str(type_)})"
            case Store(buf, idx, val):
                self.exec(f"{feed}store({self(buf)}, {self(idx)})")
                return None
            case Resize(buf, size):
                self.exec(f"{feed}resize({self(buf)}, {self(size)})")
                return None
            case Length(buf):
                return f"length({self(buf)})"
            case Block(bodies):
                ctx_2 = self.block()
                for body in bodies:
                    ctx_2(body)
                self.exec(ctx_2.emit())
                return None
            case ForLoop(var, start, end, body):
                var_2 = self(var)
                start = self(start)
                end = self(end)
                ctx_2 = self.subblock()
                ctx_2(body)
                body_code = ctx_2.emit()
                self.exec(f"{feed}for {var_2} in range({start}, {end}):\n{body_code}")
                return None
            case BufferLoop(buf, var, body):
                raise NotImplementedError
            case WhileLoop(cond, body):
                cond_code = self(cond)
                ctx_2 = self.subblock()
                ctx_2(body)
                body_code = ctx_2.emit()
                self.exec(f"{feed}while {cond_code}:\n{body_code}")
                return None
            case If(cond, body):
                cond_code = self(cond)
                ctx_2 = self.subblock()
                ctx_2(body)
                body_code = ctx_2.emit()
                self.exec(f"{feed}if {cond_code}:\n{body_code}")
                return None
            case IfElse(cond, body, else_body):
                cond_code = self(cond)
                ctx_2 = self.subblock()
                ctx_2(body)
                body_code = ctx_2.emit()
                ctx_3 = self.subblock()
                ctx_3(else_body)
                else_body_code = ctx_3.emit()
                self.exec(
                    f"{feed}if {cond_code}:\n{body_code}\n{feed}else:\n{else_body_code}"
                )
                return None
            case Function(Variable(func_name, return_t), args, body):
                ctx_2 = self.subblock()
                arg_decls = []
                for arg in args:
                    match arg:
                        case Variable(name, t):
                            arg_decls.append(f"{name}: {qual_str(t)}")
                        case _:
                            raise NotImplementedError(
                                f"Unrecognized argument type: {arg}"
                            )
                ctx_2(body)
                body_code = ctx_2.emit()
                feed = self.feed
                self.exec(
                    f"{feed}def {func_name}({', '.join(arg_decls)}) -> "
                    f"{qual_str(return_t)}:\n"
                    f"{body_code}\n"
                )
                return None
            case Return(value):
                self.exec(f"{feed}return {self(value)}")
                return None
            case Break():
                self.exec(f"{feed}break")
                return None
            case Module(funcs):
                for func in funcs:
                    if not isinstance(func, Function):
                        raise NotImplementedError(
                            f"Unrecognized function type: {type(func)}"
                        )
                    self(func)
                return None
            case Print(message, args):
                self.exec(f"{feed}print {message} {self(args)}")
                return None
            case _:
                raise NotImplementedError
