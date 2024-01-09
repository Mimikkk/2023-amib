from os import makedirs
from os.path import isdir, dirname
from pathlib import Path

from src.constants import SimDirectory

def trim_common_space_count(content: str) -> str:
  lines = content.splitlines()
  count = min([len(line) - len(line.lstrip()) for line in lines if line.strip() != ''])
  return '\n'.join(line[count:] for line in lines)

def trim_right_spaces(content: str) -> str:
  return '\n'.join(line.rstrip() for line in content.splitlines())

def remove_first_newlines(content: str) -> str:
  return content.strip('\r\n')

def append_newline(content: str) -> str:
  return content + '\n'

def prepare(content: str) -> str:
  content = trim_common_space_count(content)
  content = remove_first_newlines(content)
  content = trim_right_spaces(content)
  content = append_newline(content)
  return content

def create(sim: str, content: str):
  sim = pathof(sim)

  content = prepare(content)
  if not isdir(directory := dirname(sim)): makedirs(directory)
  with open(sim, 'w') as file: file.write(content)

def params(sim: str, content: dict[str, any]):
  sim = pathof(sim)

  str = 'sim_params:'
  for (key, value) in content.items():
    if isinstance(value, float): value = f'{value:.2f}'
    str += f'\n{key}:{value}'
  content = str

  if not isdir(directory := dirname(sim)): makedirs(directory)
  with open(sim, 'w') as file: file.write(content)

def read(sim: str) -> str:
  sim = pathof(sim)
  with open(sim, 'r') as file: return file.read()

def pathof(sim: str) -> str:
  if sim.startswith(SimDirectory): return sim
  return f"{SimDirectory}/{sim}.sim"

def nameof(sim: str) -> str:
  return Path(sim).name.replace('.sim', '')

def names() -> list[str]:
  return [nameof(name) for name in Path(SimDirectory).glob("*.sim")]

def contents() -> list[str]:
  return [read(name) for name in names()]

def entries() -> list[str, str]:
  return [(name, read(name)) for name in names()]
