import glob
from os import makedirs
from os.path import isdir, dirname
from pathlib import Path
from typing import TypeVar, Literal

from dacite import from_dict

from ..constants import ResourceDirectory
from ..serializer import json_serialize, json_deserialize, text_serialize, text_deserialize

T = TypeVar('T')
Format = Literal['json', 'text', 'gen']
def create(resource: str, content: T, *, mode: Literal['w', 'wb'] = 'w', format: Format = 'json'):
  resource = pathof(resource, format)

  if not isdir(directory := dirname(resource)): makedirs(directory)

  if format == 'json': serializer = json_serialize
  elif format == 'text' or format == 'gen': serializer = text_serialize
  else: raise ValueError(f"Unknown extension: {format}")

  with open(resource, mode) as file: file.write(serializer(content))

def read(
    resource: str,
    *,
    mode: Literal['r', 'rb'] = 'r',
    format: Format = 'json',
    model: type[T] = None
) -> T:
  resource = pathof(resource, format)

  if format == 'json': deserializer = json_deserialize
  elif format == 'text' or format == 'gen': deserializer = text_deserialize
  else: raise ValueError(f"Unknown extension: {format}")

  with open(resource, mode) as file:
    item = file.read()
  item = deserializer(item)
  if model is not None: item = from_dict(model, item)
  return item

def pathof(resource: str, format: Format) -> str:
  if resource.startswith(ResourceDirectory): return resource
  return f"{ResourceDirectory}/{resource}.{format}"

def nameof(resource: str) -> str:
  return str(Path(resource).relative_to(ResourceDirectory)).replace('.json', '').replace('.gen', '').replace('.text', '')

def names(path: str = "") -> list[str]:
  directory = ResourceDirectory
  if path: directory = f"{directory}/{path}"
  return list(map(nameof, glob.glob(f"{directory}/**/*.json", recursive=True)))

def contents() -> list[T]:
  return [read(name) for name in names()]

def entries() -> list[str, T]:
  return [(name, read(name)) for name in names()]
