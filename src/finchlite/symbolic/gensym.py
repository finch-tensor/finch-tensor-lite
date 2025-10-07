from collections.abc import Callable


class SymbolGenerator:
    counter: int = 0

    @classmethod
    def gensym(cls, name: str) -> str:
        sym = f"#{name}#{cls.counter}"
        cls.counter += 1
        return sym


_sg = SymbolGenerator()
gensym: Callable[[str], str] = _sg.gensym
