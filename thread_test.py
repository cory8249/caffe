from threading import Thread
from threading import Event

class Future(object):
    def __init__(self):
        self._ev = Event()

    def set_result(self, result):
        self._result = result
        self._ev.set()

    def set_exception(self, exc):
        self._exc = exc
        self._ev.set()

    def result(self):
        self._ev.wait()
        if hasattr(self, '_exc'):
            raise self._exc
        return self._result

def call_with_future(fn, future, args, kwargs):
    try:
        result = fn(*args, **kwargs)
        future.set_result(result)
    except Exception as exc:
        future.set_exception(exc)

def threaded(fn):
    def wrapper(*args, **kwargs):
        future = Future()
        Thread(target=call_with_future, args=(fn, future, args, kwargs)).start()
        return future
    return wrapper


class MyClass:
    def __init__(self):
        self.gg = 123
        self.yy = 88
    
    @threaded
    def get_my_value(self):
        return self.gg, self.yy

my_obj = MyClass()
fut = my_obj.get_my_value()  # this will run in a separate thread
print(fut.result())  # will block until result is computed
