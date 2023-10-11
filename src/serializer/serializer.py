import pickle
from typing import TypeVar

T = TypeVar('T')
def serialize(item: T) -> str:
  return pickle.dumps(item)

def deserialize(item: bytes) -> T:
  return pickle.loads(item)

def utf8(t: bytes) -> str:
  return t.decode('utf-8')
