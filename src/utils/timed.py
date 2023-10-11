import functools
from time import perf_counter
from contextlib import contextmanager
from asyncio import iscoroutinefunction

def timed(fn):
  @contextmanager
  def timer():
    start_ts = perf_counter()
    yield
    duration = perf_counter() - start_ts
    print(f'Fn:{fn.__name__} - {duration:.4f}[s]')

  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    if not iscoroutinefunction(fn):
      with timer():
        return fn(*args, **kwargs)

    async def coroutine():
      with timer():
        return await fn(*args, **kwargs)
    return coroutine()
  return wrapper
