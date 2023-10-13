import json
import pickle
from typing import TypeVar

T = TypeVar('T')
def pickle_serialize(content: T) -> bytes:
  return pickle.dumps(content)

def pickle_deserialize(content: bytes) -> T:
  return pickle.loads(content)

def json_serialize(content: T) -> str:
  return json.dumps(content, indent=2)

def json_deserialize(content: str) -> T:
  return json.loads(content)

def text_serialize(content: str) -> str:
  return content

def text_deserialize(content: str) -> str:
  return content
