from functools import wraps
from itertools import groupby
import os
import pickle

from matplotlib import pyplot as plt
from matplotlib.pyplot import colormaps
import numpy as np

from figures import ensure_directory
import src.resources as resources
from src.resources.models import SaveRecord

def cache_pickle(path: str, name: str = 'data'):
  def decorator(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
      if os.path.exists(path):
        print(f"Loaded {name} from {path}...")
        with open(path, 'rb') as file:
          return pickle.load(file)

      cache = fn(*args, **kwargs)
      print(f"Stored {name} in {path}...")
      with open(path, 'wb') as file:
        pickle.dump(cache, file)
      return cache
    return wrapper
  return decorator

def add(a: float, b: float) -> float: return a + b
def sub(a: float, b: float) -> float: return a - b
def sum(a: list[float], b: list[float]) -> list[float]: return list(map(add, a, b))
def dif(a: list[float], b: list[float]) -> list[float]: return list(map(sub, a, b))
def avg(a: list[float], b: float) -> list[float]: return list(map(lambda x: x / b, a))

def read_score(record: SaveRecord) -> list[float]: return [
  chapter.values[0]['vertpos']
  for chapter in record.history
]

def read_scores(records: list[SaveRecord]) -> list[list[float]]:
  return list(map(read_score, records))

def read_name(record: SaveRecord) -> str:
  return record.meta.arguments['genformat']

def groupby_representation(records: list[SaveRecord]) -> list[tuple[str, list[SaveRecord]]]: return [
  (representation, list(values))
  for (representation, values) in groupby(records, read_name)
]

def aggregate(records: list[SaveRecord], fn) -> list[float]:
  return list(map(fn, zip(*read_scores(records))))

def boxplot(records: list[SaveRecord]):
  plt.tight_layout()
  plt.figure(figsize=(12, 5))

  representations = {
    representation: [record.population[0].values['vertpos'] for record in values]
    for representation, values in groupby_representation(records)
  }

  plt.boxplot(representations.values(), labels=[f"f{k}" for k in representations.keys()])
  plt.xticks(rotation=15)
  plt.xlabel("Representation")
  plt.ylabel("Fitness")
  plt.savefig(f'resources/lab-7/figures/all-boxplot.png', bbox_inches='tight')

def plot(records: list[SaveRecord]):
  plt.tight_layout()
  plt.figure(figsize=(12, 5))

  generations = len(records[0].history)
  ticks = range(1, generations + 1)

  representations = {
    representation: list(values)
    for representation, values in groupby_representation(records)
  }

  cmap = colormaps['tab20']
  for (i, (representation, individuals)) in enumerate(representations.items()):
    for (j, individual) in enumerate(individuals):
      scores = read_score(individual)
      if j == 0:
        plt.plot(ticks, scores, label=f"f{representation}", color=cmap(i))
      else:
        plt.plot(ticks, scores, color=cmap(i))

  plt.xlabel("Generation")
  plt.ylabel("Fitness")
  plt.legend(
    loc='center left',
    bbox_to_anchor=(0.96, 0.5),
  )
  plt.savefig(f'resources/lab-7/figures/all-plot.png', bbox_inches='tight')

def aggregated(records: list[SaveRecord]):
  plt.tight_layout()
  plt.figure(figsize=(12, 5))

  generations = len(records[0].history)
  representations = {
    representation: (aggregate(values, np.mean), aggregate(values, np.std))
    for representation, values in groupby_representation(records)
  }

  ticks = range(1, generations + 1)
  for (representation, (averages, stddevs)) in representations.items():
    plt.plot(ticks, averages, label=f"f{representation}")
    stddevs = [
      stddev / 5
      for stddev in stddevs
    ]

    plt.fill_between(
      ticks,
      dif(averages, stddevs),
      sum(averages, stddevs),
      alpha=0.3
    )

  plt.ylabel('Fitness')
  plt.legend(
    loc='center left',
    bbox_to_anchor=(0.96, 0.5),
  )
  plt.savefig(f'resources/lab-7/figures/all-aggregated.png', bbox_inches='tight')

def read_records(): return [
  resources.read(name, model=SaveRecord) for name in resources.names("./lab-7/results")
]

def trim_records(records: list[SaveRecord]) -> list[SaveRecord]:
  return [
    record
    for record in records
  ]

def main():
  records = read_records()
  records = trim_records(records)

  ensure_directory(f'resources/lab-7/figures')
  boxplot(records)
  plot(records)
  aggregated(records)

if __name__ == '__main__':
  main()
