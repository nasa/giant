# Copyright 2021 United States Government as represented by the Administrator of the National Aeronautics and Space
# Administration.  No copyright is claimed in the United States under Title 17, U.S. Code. All Other Rights Reserved.


from multiprocessing.queues import Queue
from multiprocessing import Value, get_context

from queue import Empty

from typing import Any, cast


class SharedCounter:
    """
    A synchronized shared counter.

    The locking done by multiprocessing.Value ensures that only a single
    process or thread may read or write the in-memory ctypes object. However,
    in order to do n += 1, Python performs a read followed by a write, so a
    second process may read the old value before the new one is written by the
    first process. The solution is to use a multiprocessing.Lock to guarantee
    the atomicity of the modifications to Value.

    This class comes almost entirely from Eli Bendersky's blog:
    http://eli.thegreenplace.net/2012/01/04/shared-counter-with-pythons-multiprocessing/
    """

    def __init__(self, n: int = 0):
        self.count = Value('i', n)

    def increment(self, n: int = 1):
        """ Increment the counter by n (default = 1) """
        with self.count.get_lock():
            self.count.value += n

    @property
    def value(self) -> int:
        """ Return the value of the counter """
        return self.count.value


class ClearableQueue(Queue):
    """
    A portable implementation of multiprocessing.Queue.

    Because of multithreading / multiprocessing semantics, Queue.qsize() may
    raise the NotImplementedError exception on Unix platforms like Mac OS X
    where sem_getvalue() is not implemented. This subclass addresses this
    problem by using a synchronized shared counter (initialized to zero) and
    increasing / decreasing its value every time the put() and get() methods
    are called, respectively. This not only prevents NotImplementedError from
    being raised, but also allows us to implement a reliable version of both
    qsize() and empty().

    Borrowed from https://github.com/keras-team/autokeras/issues/368 and https://stackoverflow.com/a/36018632/3431189

    """

    size: SharedCounter

    def __init__(self, *args: list, **kwargs: dict):
        ctx = get_context()
        super().__init__(*args, **kwargs, ctx=ctx) # pyright: ignore[reportArgumentType]
        self.size = SharedCounter(0)

        self.holder = []

    @property
    def maxsize(self) -> int:
        
        return getattr(self, "_maxsize", -1)

    def put(self, *args, **kwargs):
        super().put(*args, **kwargs)
        self.size.increment(1)

    def get(self, *args, **kwargs) -> Any:
        """
        Gets the results and tries to flush from the holder if anything is in it
        """
        res = super().get(*args, **kwargs)
        try:
            self.size.increment(-1)
        except AttributeError:
            print('something is real wrong')

        self.flush_holder()
        return res

    def __getstate__(self):
        return cast(tuple, super().__getstate__()) + (self.size, self.holder)

    def __setstate__(self, state):
        self.size = state[-2]
        self.holder = state[-1]
        super().__setstate__(state[:-2]) # pyright: ignore[reportAttributeAccessIssue]

    def flush_holder(self):
        """
        Flushes the holder into the queue if it can be
        """

        removes = []
        for ind, held in enumerate(self.holder):
            if 0 < self.maxsize <= self.qsize():
                break
            self.put(held)

            removes.append(ind)

        for rm in removes[::-1]:
            self.holder.pop(rm)

    def get_nowait(self) -> Any:
        res = super().get_nowait()
        self.size.increment(-1)
        self.flush_holder()
        return res

    def put_nowait(self, obj: Any) -> None:
        res = super().put_nowait(obj)
        self.size.increment(1)
        return res

    def put_retry(self, item: Any):
        """
        Attempts to put a value unless the queue is full, in which case it will hold onto it until its not full and
        then put it.
        :param item: The thing to be put
        """

        self.holder.append(item)

        self.flush_holder()

    def qsize(self) -> int:
        """ Reliable implementation of multiprocessing.Queue.qsize() """
        return self.size.value + len(self.holder)

    def empty(self) -> bool:
        """ Reliable implementation of multiprocessing.Queue.empty() """
        return not self.qsize()

    def clear(self):
        """
        Clear out any data from the queue
        """

        try:
            while True:
                self.get_nowait()
        except Empty:
            pass
