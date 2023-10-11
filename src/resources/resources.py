from os import makedirs
from os.path import isdir, dirname
from pathlib import Path
from typing import TypeVar, Callable

from constants import ResourceDirectory, Serialize, Deserialize

T = TypeVar('T')
def create(resource: str, content: T, serialize: Callable[T, str] = Serialize):
  resource = pathof(resource)

  if not isdir(directory := dirname(resource)): makedirs(directory)

  with open(resource, 'wb') as file: file.write(serialize(content))

def read(resource: str, deserialize: Callable[str, T] = Deserialize) -> T:
  resource = pathof(resource)

  with open(resource, 'rb') as file: return deserialize(file.read())

def pathof(resource: str) -> str:
  if resource.startswith(ResourceDirectory): return resource
  return f"{ResourceDirectory}/{resource}.serialized"

def nameof(resource: str) -> str:
  return Path(resource).name.replace('.serialized', '')

def names() -> list[str]:
  return [nameof(name) for name in Path(ResourceDirectory).glob("*.serialized")]

def contents() -> list[T]:
  return [read(name) for name in names()]

def entries() -> list[str, T]:
  return [(name, read(name)) for name in names()]
