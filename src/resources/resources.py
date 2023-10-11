from os import makedirs
from os.path import isdir, dirname
from typing import TypeVar, Callable

from constants import SaveDirectory, Serialize, Deserialize

T = TypeVar('T')
def create(path: str, content: T, serialize: Callable[T, str] = Serialize):
  path = pathof(path)

  if not isdir(directory := dirname(path)): makedirs(directory)

  with open(path, 'wb') as file: file.write(serialize(content))

def read(path: str, deserialize: Callable[str, T] = Deserialize) -> T:
  path = pathof(path)

  with open(path, 'rb') as file: return deserialize(file.read())

def pathof(path: str) -> str:
  return f"{SaveDirectory}/{path}.serialized"

def listed() -> list[str]:
  from os import listdir
  from os.path import isfile, join

  return [file for file in listdir(SaveDirectory) if isfile(join(SaveDirectory, file))]

def contents() -> list[T]:
  return [read(file) for file in listed()]
