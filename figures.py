import os

def ensure_directory(path: str):
  if os.path.exists(path): return
  os.makedirs(path)
