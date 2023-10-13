from os import makedirs
from os.path import isdir, dirname
from pathlib import Path
from typing import TypeVar, Literal

from constants import ResourceDirectory
from serializer import json_serialize, json_deserialize, text_serialize, text_deserialize

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
) -> T:
  resource = pathof(resource, format)

  if format == 'json': deserializer = json_deserialize
  elif format == 'text' or format == 'gen': deserializer = text_deserialize
  else: raise ValueError(f"Unknown extension: {format}")

  with open(resource, mode) as file: return deserializer(file.read())

def pathof(resource: str, format: Format) -> str:
  if resource.startswith(ResourceDirectory): return resource
  return f"{ResourceDirectory}/{resource}.{format}"

def nameof(resource: str) -> str:
  return Path(resource).name.replace('.json', '').replace('.gen', '').replace('.text', '')

def names() -> list[str]:
  return [nameof(name) for name in Path(ResourceDirectory).glob("*.json|*.gen|*.text")]

def contents() -> list[T]:
  return [read(name) for name in names()]

def entries() -> list[str, T]:
  return [(name, read(name)) for name in names()]
