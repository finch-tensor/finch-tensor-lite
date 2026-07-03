"""Device policies, execution pools, and tasks for Finch metadata."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import cpu_count
from typing import Any

from .ftypes import FType, FTyped, ftype


def _default_num_tasks() -> int:
    return cpu_count() or 1


class PoolFType(FType, ABC):
    @property
    @abstractmethod
    def num_tasks(self):
        ...

    @property
    @abstractmethod
    def pool(self):
        ...


class DeviceFType(FType, ABC):
    @property
    @abstractmethod
    def pool_type(self) -> PoolFType:
        ...

    @property
    @abstractmethod
    def parent_device_type(self):
        ...


class TaskFType(FType, ABC):
    @property
    @abstractmethod
    def num_tasks(self):
        ...

    @property
    @abstractmethod
    def task_num(self):
        ...

    @property
    @abstractmethod
    def pool(self):
        ...

    @property
    @abstractmethod
    def parent_task(self):
        ...


class AbstractPool(FTyped, ABC):
    @property
    @abstractmethod
    def num_tasks(self):
        ...

    @property
    @abstractmethod
    def pool(self):
        ...


class AbstractDevice(FTyped, ABC):
    @property
    @abstractmethod
    def pool(self) -> AbstractPool:
        ...

    @property
    @abstractmethod
    def parent_device(self):
        ...

    @property
    def num_tasks(self):
        return self.pool.num_tasks


class AbstractTask(FTyped, ABC):
    @property
    @abstractmethod
    def num_tasks(self):
        ...

    @property
    @abstractmethod
    def task_num(self):
        ...

    @property
    @abstractmethod
    def pool(self):
        ...

    @property
    @abstractmethod
    def parent_task(self):
        ...

    def is_on_pool(self, pool: AbstractPool) -> bool:
        pool = normalize_pool(pool)
        task = self
        while task is not None:
            if task.pool == pool:
                return True
            task = task.parent_task
        return False


@dataclass(frozen=True, slots=True)
class SerialPool(AbstractPool):
    @property
    def ftype(self):
        return SerialPoolFType()

    @property
    def num_tasks(self):
        return 1

    @property
    def pool(self):
        return CPUPool(1)


def serial_pool() -> SerialPool:
    return SerialPool()


@dataclass(frozen=True, slots=True, eq=False)
class CPUPool(AbstractPool):
    n: int
    id: Any = "default"

    def __post_init__(self):
        if self.n < 1:
            raise ValueError(f"CPU pool requires at least one task, got {self.n}")

    @property
    def ftype(self):
        return CPUPoolFType(self.id)

    @property
    def num_tasks(self):
        return self.n

    @property
    def pool(self):
        return self

    def __eq__(self, other):
        match other:
            case CPUPool(id=other_id):
                return self.id == other_id
            case _:
                return False

    def __hash__(self):
        return hash((CPUPool, self.id))


def cpu_pool(id: Any = "default", n: int | None = None) -> CPUPool:
    return CPUPool(_default_num_tasks() if n is None else n, id)


@dataclass(frozen=True, slots=True)
class Serial(AbstractDevice):
    @property
    def ftype(self):
        return SerialFType()

    @property
    def pool(self):
        return serial_pool()

    @property
    def parent_device(self):
        return None

    def __repr__(self):
        return "Serial"


def serial() -> Serial:
    return Serial()


def _normalize_parent_device(parent):
    match parent:
        case None:
            return serial()
        case type() if parent is Serial:
            return serial()
        case AbstractDevice():
            return parent
        case _:
            raise ValueError(f"device parent is not supported; got {parent!r}")


@dataclass(frozen=True, slots=True, eq=False, init=False)
class CPU(AbstractDevice):
    __match_args__ = ("parent", "id")

    parent: AbstractDevice
    n: int | None
    id: Any

    def __init__(self, parent=None, /, id: Any = "default", n: int | None = None):
        if isinstance(parent, int):
            if n is not None:
                raise TypeError("CPU received task count twice")
            n = parent
            parent = serial()
        if n is not None and n < 1:
            raise ValueError(f"CPU device requires at least one task, got {n}")
        object.__setattr__(self, "parent", _normalize_parent_device(parent))
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "id", id)

    @property
    def ftype(self):
        return CPUFType(self.parent.ftype, self.id)

    @property
    def pool(self):
        return CPUPool(_default_num_tasks() if self.n is None else self.n, self.id)

    @property
    def parent_device(self):
        return self.parent

    def __eq__(self, other):
        match other:
            case CPU(parent=other_parent, id=other_id):
                return self.parent == other_parent and self.id == other_id
            case _:
                return False

    def __hash__(self):
        return hash((CPU, self.parent, self.id))

    def __repr__(self):
        suffix = "" if self.id == "default" else f", id={self.id!r}"
        if self.n is not None:
            suffix += f", n={self.n!r}"
        return f"CPU({self.parent!r}{suffix})"


def cpu(id: Any = "default", n: int | None = None, parent=None) -> CPU:
    return CPU(parent, id=id, n=n)


@dataclass(frozen=True, slots=True)
class SerialTask(AbstractTask):
    @property
    def ftype(self):
        return SerialTaskFType()

    @property
    def num_tasks(self):
        return 1

    @property
    def task_num(self):
        return 1

    @property
    def pool(self):
        return serial_pool()

    @property
    def parent_task(self):
        return None


@dataclass(frozen=True, slots=True, init=False)
class CPUThread(AbstractTask):
    __match_args__ = ("tid", "pool", "parent")

    tid: int
    _pool: CPUPool
    parent: AbstractTask | None = None

    def __init__(
        self, tid: int, pool: CPUPool | CPU, parent: AbstractTask | None = None
    ):
        match pool:
            case CPU():
                pool = pool.pool
            case CPUPool():
                pass
            case _:
                raise ValueError(f"CPUThread pool is not supported; got {pool!r}")
        object.__setattr__(self, "tid", tid)
        object.__setattr__(self, "_pool", pool)
        object.__setattr__(self, "parent", parent)

    @property
    def ftype(self):
        parent_type = (
            ftype(self.parent_task) if self.parent_task is not None else ftype(None)
        )
        return CPUThreadFType(parent_type, self.pool.ftype)

    @property
    def num_tasks(self):
        return self.pool.num_tasks

    @property
    def task_num(self):
        return self.tid

    @property
    def pool(self):
        return self._pool

    @property
    def parent_task(self):
        return self.parent

    def __repr__(self):
        return (
            f"CPUThread(tid={self.tid!r}, "
            f"pool={self.pool!r}, parent={self.parent!r})"
        )


@dataclass(frozen=True, slots=True, eq=False)
class SerialPoolFType(PoolFType):
    def __eq__(self, other):
        match other:
            case SerialPoolFType():
                return True
            case _:
                return False

    def __hash__(self):
        return hash(SerialPoolFType)

    def __call__(self, *args):
        if args:
            raise TypeError("SerialPoolFType expects no arguments")
        return SerialPool()

    @property
    def num_tasks(self):
        return 1

    @property
    def pool(self):
        return CPUPoolFType()

    def __repr__(self):
        return "SerialPool"


@dataclass(frozen=True, slots=True, eq=False)
class CPUPoolFType(PoolFType):
    id: Any = "default"

    def __eq__(self, other):
        match other:
            case CPUPoolFType(id=other_id):
                return self.id == other_id
            case _:
                return False

    def __hash__(self):
        return hash((CPUPoolFType, self.id))

    def __call__(self, n: int | None = None):
        return CPUPool(_default_num_tasks() if n is None else n, self.id)

    @property
    def num_tasks(self):
        return _default_num_tasks()

    @property
    def pool(self):
        return self

    def __repr__(self):
        return f"CPUPool{{{self.id!r}}}"


@dataclass(frozen=True, slots=True, eq=False)
class SerialFType(DeviceFType):
    def __eq__(self, other):
        match other:
            case SerialFType():
                return True
            case _:
                return False

    def __hash__(self):
        return hash(SerialFType)

    def __call__(self, *args):
        if args:
            raise TypeError("SerialFType expects no arguments")
        return Serial()

    @property
    def pool_type(self):
        return SerialPoolFType()

    @property
    def parent_device_type(self):
        return None

    def __repr__(self):
        return "Serial"


@dataclass(frozen=True, slots=True, eq=False, init=False)
class CPUFType(DeviceFType):
    parent_type: DeviceFType
    id: Any

    def __init__(self, parent_type=None, id: Any = "default"):
        if parent_type is None:
            parent_type = SerialFType()
        elif not isinstance(parent_type, DeviceFType):
            if id != "default":
                raise TypeError("CPUFType received id twice")
            id = parent_type
            parent_type = SerialFType()
        object.__setattr__(self, "parent_type", parent_type)
        object.__setattr__(self, "id", id)

    def __eq__(self, other):
        match other:
            case CPUFType(parent_type=parent_type, id=other_id):
                return self.parent_type == parent_type and self.id == other_id
            case _:
                return False

    def __hash__(self):
        return hash((CPUFType, self.parent_type, self.id))

    def __call__(self, parent=None, n: int | None = None):
        if parent is None:
            parent = self.parent_type()
        return CPU(parent, id=self.id, n=n)

    @property
    def pool_type(self):
        return CPUPoolFType(self.id)

    @property
    def parent_device_type(self):
        return self.parent_type

    def __repr__(self):
        suffix = "" if self.id == "default" else f", id={self.id!r}"
        return f"CPU({self.parent_type!r}{suffix})"


@dataclass(frozen=True, slots=True, eq=False)
class SerialTaskFType(TaskFType):
    def __eq__(self, other):
        match other:
            case SerialTaskFType():
                return True
            case _:
                return False

    def __hash__(self):
        return hash(SerialTaskFType)

    def __call__(self, *args):
        if args:
            raise TypeError("SerialTaskFType expects no arguments")
        return SerialTask()

    @property
    def num_tasks(self):
        return 1

    @property
    def task_num(self):
        return 1

    @property
    def pool(self):
        return SerialPoolFType()

    @property
    def parent_task(self):
        return None


@dataclass(frozen=True, slots=True, eq=False)
class CPUThreadFType(TaskFType):
    parent_type: FType
    pool_type: CPUPoolFType

    def __eq__(self, other):
        match other:
            case CPUThreadFType(parent_type=parent_type, pool_type=pool_type):
                return self.parent_type == parent_type and self.pool_type == pool_type
            case _:
                return False

    def __hash__(self):
        return hash((CPUThreadFType, self.parent_type, self.pool_type))

    def __call__(self, tid: int, pool: CPUPool | CPU, parent=None):
        return CPUThread(tid, pool, parent)

    @property
    def num_tasks(self):
        return self.pool_type.num_tasks

    @property
    def task_num(self):
        raise TypeError("CPUThreadFType does not carry a task number")

    @property
    def pool(self):
        return self.pool_type

    @property
    def parent_task(self):
        return self.parent_type


def normalize_pool(pool: Any) -> AbstractPool:
    if pool is None:
        return serial_pool()
    match pool:
        case AbstractDevice():
            return pool.pool
        case AbstractPool():
            return pool
        case _:
            raise ValueError(f"pool argument is not supported; got {pool!r}")


def normalize_device(device: Any) -> AbstractDevice:
    if device is None:
        return serial()
    match device:
        case type() if device is Serial:
            return serial()
        case AbstractDevice():
            return device
        case _:
            raise ValueError(f"device argument is not supported; got {device!r}")


def is_parent_device(parent: Any, child: Any) -> bool:
    parent = normalize_device(parent)
    child = normalize_device(child)
    while child is not None:
        if child == parent:
            return True
        child = child.parent_device
    return False


def common_device(*devices: Any) -> AbstractDevice:
    if not devices:
        return serial()
    device = normalize_device(devices[0])
    for other in devices[1:]:
        other = normalize_device(other)
        if is_parent_device(device, other):
            device = other
        elif not is_parent_device(other, device):
            raise ValueError("Inputs must be on compatible devices")
    return device


def is_on_pool(task: Any, pool: AbstractPool) -> bool:
    return task.is_on_pool(pool)


def is_on_device(task: Any, dev: AbstractPool | AbstractDevice) -> bool:
    return is_on_pool(task, dev)
