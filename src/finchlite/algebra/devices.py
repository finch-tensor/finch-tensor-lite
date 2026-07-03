"""Device policies and execution tasks for Finch metadata."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import cpu_count
from typing import Any, cast

from .ftypes import FType, FTyped, ftype


def _default_num_tasks() -> int:
    return cpu_count() or 1


class DeviceFType(FType, ABC):
    @property
    @abstractmethod
    def num_tasks(self):
        ...

    @property
    @abstractmethod
    def device(self):
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
    def device(self):
        ...

    @property
    @abstractmethod
    def parent_task(self):
        ...


class AbstractDevice(FTyped, ABC):
    @property
    @abstractmethod
    def num_tasks(self):
        ...

    @property
    @abstractmethod
    def device(self):
        ...

    @property
    @abstractmethod
    def parent_device(self):
        ...


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
    def device(self):
        ...

    @property
    @abstractmethod
    def parent_task(self):
        ...

    def is_on_device(self, device: Any) -> bool:
        device = normalize_device(device)
        task = self
        while task is not None:
            if task.device == device:
                return True
            task = task.parent_task
        return False


@dataclass(frozen=True, slots=True)
class Serial(AbstractDevice):
    @property
    def ftype(self):
        return SerialFType()

    @property
    def num_tasks(self):
        return 1

    @property
    def device(self):
        return self

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
    def num_tasks(self):
        return _default_num_tasks() if self.n is None else self.n

    @property
    def device(self):
        return self

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
    def device(self):
        return serial()

    @property
    def parent_task(self):
        return None


@dataclass(frozen=True, slots=True, init=False)
class CPUThread(AbstractTask):
    __match_args__ = ("tid", "device", "parent")

    tid: int
    _device: CPU
    parent: AbstractTask | None = None

    def __init__(self, tid: int, device: CPU, parent: AbstractTask | None = None):
        match device:
            case CPU():
                pass
            case _:
                raise ValueError(f"CPUThread device is not supported; got {device!r}")
        object.__setattr__(self, "tid", tid)
        object.__setattr__(self, "_device", device)
        object.__setattr__(self, "parent", parent)

    @property
    def ftype(self):
        parent_type = (
            ftype(self.parent_task) if self.parent_task is not None else ftype(None)
        )
        return CPUThreadFType(parent_type, self.device.ftype)

    @property
    def num_tasks(self):
        return self.device.num_tasks

    @property
    def task_num(self):
        return self.tid

    @property
    def device(self):
        return self._device

    @property
    def parent_task(self):
        return self.parent

    def __repr__(self):
        return (
            f"CPUThread(tid={self.tid!r}, "
            f"device={self.device!r}, parent={self.parent!r})"
        )


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
    def num_tasks(self):
        return 1

    @property
    def device(self):
        return self

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
            parent = cast(Any, self.parent_type)()
        return CPU(parent, id=self.id, n=n)

    @property
    def num_tasks(self):
        return _default_num_tasks()

    @property
    def device(self):
        return self

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
    def device(self):
        return SerialFType()

    @property
    def parent_task(self):
        return None


@dataclass(frozen=True, slots=True, eq=False)
class CPUThreadFType(TaskFType):
    parent_type: FType
    device_type: CPUFType

    def __eq__(self, other):
        match other:
            case CPUThreadFType(parent_type=parent_type, device_type=device_type):
                return (
                    self.parent_type == parent_type
                    and self.device_type == device_type
                )
            case _:
                return False

    def __hash__(self):
        return hash((CPUThreadFType, self.parent_type, self.device_type))

    def __call__(self, tid: int, device: CPU, parent=None):  # type: ignore[override]
        return CPUThread(tid, device, parent)

    @property
    def num_tasks(self):
        return self.device_type.num_tasks

    @property
    def task_num(self):
        raise TypeError("CPUThreadFType does not carry a task number")

    @property
    def device(self):
        return self.device_type

    @property
    def parent_task(self):
        return self.parent_type


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


def is_on_device(task: Any, dev: Any) -> bool:
    return task.is_on_device(dev)
