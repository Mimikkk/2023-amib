from os import makedirs
from os.path import isdir, dirname
from pathlib import Path
from constants import SimDirectory

def trim_common_space_count(content: str) -> str:
  lines = content.splitlines()
  common_space_count = min([len(line) - len(line.lstrip()) for line in lines if line.strip() != ''])
  trimmed_lines = [line[common_space_count:] for line in lines]
  return '\n'.join(trimmed_lines)

def remove_first_newlines(content: str) -> str:
  return content.lstrip('\r\n')

def prepare(content: str) -> str:
  content = trim_common_space_count(content)
  content = remove_first_newlines(content)
  return content

def create(sim: str, content: str):
  sim = pathof(sim)

  content = prepare(content)
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
