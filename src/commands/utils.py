import multiprocessing
from typing import overload, Iterable

from .command import Command

@overload
def invoke(command: Command): ...

@overload
def invoke(commands: Iterable[Command]): ...

def invoke(item: Command | Iterable[Command]):
  if isinstance(item, Command):
    process = multiprocessing.Process(target=item.run)
    process.start()
    process.join()
    return

  processes = [multiprocessing.Process(target=command.run) for command in item]
  for process in processes: process.start()
  for process in processes: process.join()
