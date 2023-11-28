from itertools import groupby

from matplotlib import pyplot as plt
from matplotlib.pyplot import colormaps
import numpy as np

from figures import ensure_directory
import resources
from resources.models import SaveRecord


def add(a: float, b: float) -> float: return a + b
def sub(a: float, b: float) -> float: return a - b
def sum(a: list[float], b: list[float]) -> list[float]: return list(map(add, a, b))
def dif(a: list[float], b: list[float]) -> list[float]: return list(map(sub, a, b))
def avg(a: list[float], b: float) -> list[float]: return list(map(lambda x: x / b, a))

def read_score(record: SaveRecord) -> list[float]: return [
  chapter.values[0]['vertpos']
  if 'vertpos' in chapter.values[0] else
  chapter.values[0]['numparts']
  for chapter in record.history
]

def read_scores(records: list[SaveRecord]) -> list[list[float]]:
  return list(map(read_score, records))

def label_by_prefix() -> str:
  match prefix:
    case "1": return "vp"
    case "2": return "np"
    case "3": return "np+vp"
    case "4": return "vp+np-cond"

def read_name(record: SaveRecord) -> str:
  representation = record.name.split('-')[-2]
  prefix = record.name.split('/')[1]

  label = ""
  match prefix:
    case "1": label = "vp"
    case "2": label = "np"
    case "3": label = "np+vp"
    case "4": label = "vp+np|cond"

  return f"{label}-{representation}"

def groupby_representation(records: list[SaveRecord]): return [
  (representation, list(values))
  for (representation, values) in groupby(records, read_name)
]

def aggregate(records: list[SaveRecord], fn) -> list[float]:
  return list(map(fn, zip(*read_scores(records))))

prefix = '4'

def boxplot(first: list[SaveRecord]):
  plt.tight_layout()
  plt.figure(figsize=(12, 5))

  representations = {
    representation: aggregate(values, np.mean)
    for representation, values in groupby_representation(first)
  }

  plt.boxplot(representations.values(), labels=representations.keys())
  plt.xticks(rotation=15)
  plt.xlabel("Representation")
  plt.ylabel("Fitness")
  plt.savefig(f'resources/lab-2/{prefix}/figures/{label_by_prefix()}-boxplot.png', bbox_inches='tight')

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
        plt.plot(ticks, scores, label=representation, color=cmap(i))
      else:
        plt.plot(ticks, scores, color=cmap(i))

  plt.xlabel("Generation")
  plt.ylabel("Fitness")
  plt.legend(
    loc='center left',
    bbox_to_anchor=(0.96, 0.5),
  )
  plt.savefig(f'resources/lab-2/{prefix}/figures/{label_by_prefix()}-plot.png', bbox_inches='tight')

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
    plt.plot(ticks, averages, label=representation)
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
  plt.savefig(f'resources/lab-2/{prefix}/figures/{label_by_prefix()}-aggregated.png', bbox_inches='tight')


def main():
  records = [resources.read(name, model=SaveRecord) for name in resources.names(f'lab-2/{prefix}')]

  ensure_directory(f'resources/lab-2/{prefix}/figures')
  boxplot(records)
  plot(records)
  aggregated(records)

if __name__ == '__main__':
  main()
