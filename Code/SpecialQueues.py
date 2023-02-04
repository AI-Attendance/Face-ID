from multiprocessing import Condition, Event, Pipe, Lock, RLock, Value


class SignalingQueue:

    def __init__(self, MultipleReaders=False, MultipleWriters=False):
        self._read_lock = Lock() if MultipleReaders else None
        self._write_lock = Lock() if MultipleWriters else None
        self._read_end, self._write_end = Pipe(duplex=False)
        self._data_count = Value("I", 0, lock=False)
        self._data_count_lock = Lock()
        self._event = Event()
        self._condition = Condition(lock=Lock())

    def _guard_execution(self, lock, action, *args, **kwargs):
        if lock is not None:
            lock.acquire()
        ret = action(*args, **kwargs)
        if lock is not None:
            lock.release()
        return ret

    def _increment_count(self) -> None:
        self._data_count.value += 1

    def put(self, dataToSend) -> bool:
        if self.signaled():
            return False
        self._guard_execution(self._write_lock, self._write_end.send,
                              dataToSend)
        self._guard_execution(self._data_count_lock, self._increment_count)
        self._guard_execution(self._condition, self._condition.notify,
                              self._data_count.value)
        return True

    def _decrement_count(self) -> None:
        self._data_count.value -= 1

    def get(self):
        self._guard_execution(self._condition,
                              self._condition.wait_for,
                              predicate=lambda: self.signaled() or self.any())
        if self.empty():
            return None
        self._guard_execution(self._data_count_lock, self._decrement_count)
        return self._guard_execution(self._read_lock, self._read_end.recv)

    def signal(self) -> None:
        if self.signaled():
            return
        self._event.set()
        self._guard_execution(self._condition, self._condition.notify_all)

    def signaled(self) -> bool:
        return self._event.is_set()

    def qsize(self) -> int:
        return self._data_count.value

    def any(self) -> bool:
        return bool(self._data_count.value)

    def empty(self) -> bool:
        return not self._data_count.value
