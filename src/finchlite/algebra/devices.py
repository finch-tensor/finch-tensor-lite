"""Device and task hierarchy for Finch execution metadata."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from os import cpu_count
from typing import Any

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

    def is_on_device(self, dev: AbstractDevice) -> bool:
        task = self
        while task is not None:
            if task.device == dev:
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
        return CPU(1)


def serial() -> Serial:
    return Serial()


@dataclass(frozen=True, slots=True, eq=False)
class CPU(AbstractDevice):
    n: int
    id: Any = "default"

    def __post_init__(self):
        if self.n < 1:
            raise ValueError(f"CPU requires at least one task, got {self.n}")

    @property
    def ftype(self):
        return CPUFType(self.id)

    @property
    def num_tasks(self):
        return self.n

    @property
    def device(self):
        return self

    def __eq__(self, other):
        match other:
            case CPU(id=other_id):
                return self.id == other_id
            case _:
                return False

    def __hash__(self):
        return hash((CPU, self.id))


def cpu(id: Any = "default", n: int | None = None) -> CPU:
    return CPU(_default_num_tasks() if n is None else n, id)


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
        return Serial()

    @property
    def parent_task(self):
        return None


@dataclass(frozen=True, slots=True, init=False)
class CPUThread(AbstractTask):
    tid: int
    _device: CPU
    parent: AbstractTask | None = None

    def __init__(self, tid: int, device: CPU, parent: AbstractTask | None = None):
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
        return CPUFType()

    def __repr__(self):
        return "Serial"


@dataclass(frozen=True, slots=True, eq=False)
class CPUFType(DeviceFType):
    id: Any = "default"

    def __eq__(self, other):
        match other:
            case CPUFType(id=other_id):
                return self.id == other_id
            case _:
                return False

    def __hash__(self):
        return hash((CPUFType, self.id))

    def __call__(self, n: int | None = None):
        return CPU(_default_num_tasks() if n is None else n, self.id)

    @property
    def num_tasks(self):
        return _default_num_tasks()

    @property
    def device(self):
        return self

    def __repr__(self):
        return f"CPU{{{self.id!r}}}"


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

    def __call__(self, tid: int, device: CPU, parent=None):
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
        case AbstractDevice():
            return device
        case _:
            raise ValueError(f"device argument is not supported; got {device!r}")


def is_on_device(task: Any, dev: AbstractDevice) -> bool:
    return task.is_on_device(dev)
