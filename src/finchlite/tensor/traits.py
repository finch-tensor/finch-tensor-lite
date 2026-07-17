from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from finchlite.algebra import Tensor


class AccessCapability:
    pass


class Sequential(AccessCapability):
    pass


class Random(AccessCapability):
    pass


class DataProperty:
    pass


class Dense(DataProperty):
    pass


class Sparse(DataProperty):
    pass


class Blocked(DataProperty):
    pass


class Repeated(DataProperty):
    pass


class Extruded(DataProperty):
    pass


class Hollow(DataProperty):
    pass

