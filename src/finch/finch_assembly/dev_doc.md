# Finch Assembly

FinchAssembly is the final intermediate representation before code generation.
It is a low-level imperative description of the program with control flow, linear memory regions called "buffers" and explicit memory management.[^1]

## Grammar

The following is a rough grammar for FinchAssembly, written in terms of the current `__repr__`s of the corresponding AssemblyNodes.

```
EXPR       := LITERAL | VARIABLE | SLOT | STACK | GETATTR | CALL | LOAD | LENGTH
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
LOAD       := Load(buffer=SLOT | STACK, index=EXPR)
STORE      := Store(buffer=SLOT | STACK, index=EXPR, value=EXPR)
RESIZE     := Resize(buffer=SLOT | STACK, new_size=EXPR)
LENGTH     := Length(buffer=SLOT | STACK)
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

## Notes

* Assignment statements is also used for declaration.
  This means it's not possible to shadow a variable in a given scope and, in particular, if a variable already appears in the current scope then it must be assigned a value of the type given in the context.
  It is, however, possible to shadow a variable that appears in a parent scope.
* There is no higher-order programming.
  In particular, function definitions must appear at the top-level within a module.
* The bodies of loops and conditionals are not required to be blocks.
  A new scope is implicitly opened when evaluating the body of a loop or a conditional.
* At the moment, nodes in modules are required to be function definitions.
* At the moment, functions in

## Questions

* Are function monomorphic? I presume yes (in particular, variables *must* be type annotated in function definitions)
* What can the right-hand side of an `UNPACK` be? Can it be anything but a variable? If not, what does it mean to make sure the original object is not accessed or modified?

[^1]: Nathan: Not my own words, taken from docstring.
