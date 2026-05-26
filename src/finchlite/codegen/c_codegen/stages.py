from finchlite.symbolic import Stage


class CCode:
    def __init__(self, code: str):
        self.code = code

    def __str__(self) -> str:
        return self.code


class CLowerer(Stage):
    pass


__all__ = ["CCode", "CLowerer"]
