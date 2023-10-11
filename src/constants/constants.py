from typing import Callable, TypeVar

from serializer.serializer import serialize, deserialize

T = TypeVar('T')
Python: str = '../venv/Scripts/python.exe'
Library: str = 'src/libs/Framsticks50rc29'
Runfile: str = 'src/libs/framspy/FramsticksEvolution.py'
SaveDirectory: str = './resources'
Serialize: Callable[T, str] = serialize
Deserialize: Callable[str, T] = deserialize
