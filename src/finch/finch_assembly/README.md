# Finch Assembly

FinchAssembly is the final intermediate representation before code generation.
It is a low-level imperative description of the program with control flow, linear memory regions called "buffers" and explicit memory management.

## Grammar

```
EXPR       := LITERAL | VARIABLE | SLOT | GETATTR | CALL | LOAD | LENGTH
STMT       := UNPACK | REPACK | ASSIGN | SETATTR | STORE | RESIZE | FORLOOP
            | BUFFERLOOP | WHILELOOP | IF | IFELSE | FUNCTION | RETURN | BREAK
            | BLOCK | MODULE | EXPR
MODULE     := FUNCTION...

LITERAL    := Literal(val=VALUE)
VARIABLE   := Variable(name=IDENT, type=TYPE)
SLOT       := Slot(name=IDENT, type=TYPE)
UNPACK     := Unpack(lhs=SLOT, rhs=EXPR)
REPACK     := Repack(val=SLOT)
ASSIGN     := Assign(lhs=VARIABLE, rhs=EXPR)
GETATTR    := GetAttr(obj=EXPR, attr=LITERAL)
SETATTR    := SetAttr(obj=EXPR, attr=LITERAL, value=EXPR)
CALL       := Call(op=LITERAL, args=EXPR...) # NOT LITERAL???
LOAD       := Load(buffer=EXPR, index=EXPR)
STORE      := Store(buffer=EXPR, index=EXPR, value=EXPR)
RESIZE     := Resize(buffer=EXPR, new_size=EXPR)
LENGTH     := Length(buffer=EXPR)
FORLOOP    := ForLoop(var=VARIABLE, start=EXPR, end=EXPR, body=BLOCK)
BUFFERLOOP := BufferLoop(buffer=EXPR, var=VARIABLE, body=BLOCK)
WHILELOOP  := WhileLoop(condition=EXPR, body=BLOCK)
IF         := If(condition=EXPR, body=BLOCK)
IFELSE     := IfElse(condition=EXPR, body=BLOCK, else_body=BLOCK)
FUNCTION   := Function(name=VARIABLE, args=VARIABLE..., body=BLOCK)
RETURN     := Return(arg=EXPR)
BREAK      := Break()
BLOCK      := Block(bodies=STMTS...)
```
