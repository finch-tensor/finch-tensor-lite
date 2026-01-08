from collections.abc import Callable


class SymbolGenerator:
    counter: int = 0

    @classmethod
    def gensym(cls, name: str, sep: str = "#") -> str:
        sym = f"{sep}{name}{sep}{cls.counter}"
        cls.counter += 1
        return sym


_sg = SymbolGenerator()
gensym: Callable[..., str] = _sg.gensym
