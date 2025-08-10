# Finch Assembly

FinchAssembly is the final intermediate representation before code generation.
It is a low-level imperative description of the program with control flow, linear memory regions called "buffers" and explicit memory management.[^1]

## Grammar

The following is a rough grammar for FinchAssembly, written in terms of the current `__repr__`s of the corresponding AssemblyNodes.

```
EXPR       := LITERAL | VARIABLE | SLOT | GETATTR | CALL | LOAD | LENGTH | STACK
STMT       := UNPACK | REPACK | ASSIGN | SETATTR | STORE | RESIZE | FORLOOP
            | BUFFERLOOP | WHILELOOP | IF | IFELSE | FUNCTION | RETURN | BREAK
            | BLOCK | MODULE
NODE       := EXPR | STMT

LITERAL    := Literal(val=VALUE)
VARIABLE   := Variable(name=IDENT, type=TYPE)
SLOT       := Slot(name=IDENT, type=TYPE)
UNPACK     := Unpack(lhs=SLOT, rhs=EXPR)
REPACK     := Repack(val=SLOT)
ASSIGN     := Assign(lhs=VARIABLE | STACK, rhs=EXPR)
GETATTR    := GetAttr(obj=EXPR, attr=LITERAL)
SETATTR    := SetAttr(obj=EXPR, attr=LITERAL, value=EXPR)
CALL       := Call(op=LITERAL, args=EXPR...)
LOAD       := Load(buffer=EXPR, index=EXPR)
STORE      := Store(buffer=EXPR, index=EXPR, value=EXPR)
RESIZE     := Resize(buffer=EXPR, new_size=EXPR)
LENGTH     := Length(buffer=EXPR)
STACK      := Stack(obj=ANY, type=TYPE)
FORLOOP    := ForLoop(var=VARIABLE, start=EXPR, end=EXPR, body=NODE)
BUFFERLOOP := BufferLoop(buffer=EXPR, var=VARIABLE, body=NODE)
WHILELOOP  := WhileLoop(condition=EXPR, body=NODE)
IF         := If(condition=EXPR, body=NODE)
IFELSE     := IfElse(condition=EXPR, body=NODE, else_body=NODE)
FUNCTION   := Function(name=VARIABLE, args=VARIABLE..., body=NODE)
RETURN     := Return(arg=EXPR)
BREAK      := Break()
BLOCK      := Block(bodies=NODE...)
MODULE     := Module(funcs=NODE...)
```

## Questions

* What can the right-hand side of an `UNPACK` be? Can it be anything but a variable? If not, what does it mean to make sure the original object is not accessed or modified?
* Is `ASSIGN` also used for declaration? In particular, is it possible assign a variable to be something with type other than the variable's type in the given context (this seems to be the case in the interpreter)?
* I presume no higher-order programming?
* Can `UNPACK` and `REPACK` appear loops or conditionals? (I presume no)
* Can `FUNCTION` appear in loops, conditionals or other functions? (I presume no)
* Are bodies of loops, conditions, and functions required to be `BLOCK`s? Should this be reflected in signature of the constructor?
* Are the nodes in a module required to be `FUNCTION`s? Should this be reflected in the signature of the constructor?
* The left-hand side of `UNPACK` and the argument of `REPACK` seem to be allowed to be `STACK`. Should this be reflected in the signature of the constructor?

[^1]: Nathan: Not my own words, taken from docstring.
