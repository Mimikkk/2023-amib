from os import makedirs
from os.path import isdir, dirname
from typing import TypeVar, Callable

from constants import SaveDirectory, Serialize, Deserialize

T = TypeVar('T')
def create(path: str, content: T, serialize: Callable[T, str] = Serialize):
  path = f"{SaveDirectory}/{path}.serialized"

  if not isdir(directory := dirname(path)): makedirs(directory)

  with open(path, 'wb') as file: file.write(serialize(content))

def read(path: str, deserialize: Callable[str, T] = Deserialize) -> T:
  path = f"{SaveDirectory}/{path}.serialized"

  with open(path, 'rb') as file: return deserialize(file.read())
