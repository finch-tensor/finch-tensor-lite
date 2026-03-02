class SequentialRead:
    """
    Forward only iteration. No random seek.
    """


class SequentialWrite:
    """
    Accepts values written in non-decreasing order. No random writes.
    """


class RandomRead(SequentialRead):
    """
    Supports O(1) access for any index / key.
    Falls back to sequential read as a last resort.
    """


class RandomWrite:
    """
    Supports O(1) write for any index / key.
    Falls back to sequential write as a last resort.
    """


class Capabilities:
    def __init__(self, read=None, write=None):
        self.read = read
        self.write = write

    def supports(self, trait) -> bool:
        if issubclass(trait, (SequentialRead, RandomRead)):
            return self.read is not None and issubclass(self.read, trait)
        if issubclass(trait, SequentialWrite, RandomWrite):
            return self.write is not None and issubclass(self.write, trait)
        return False
