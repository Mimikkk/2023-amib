import pickle
from typing import TypeVar

T = TypeVar('T')
def serialize(item: T) -> str:
  return pickle.dumps(item)

def deserialize(item: str) -> T:
  return pickle.loads(item)
