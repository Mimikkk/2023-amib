from itertools import groupby

from matplotlib import pyplot as plt
from matplotlib.pyplot import colormaps
import numpy as np

from figures import ensure_directory
import src.resources as resources
from src.resources.models import SaveRecord


def add(a: float, b: float) -> float: return a + b
def sub(a: float, b: float) -> float: return a - b
def sum(a: list[float], b: list[float]) -> list[float]: return list(map(add, a, b))
def dif(a: list[float], b: list[float]) -> list[float]: return list(map(sub, a, b))
def avg(a: list[float], b: float) -> list[float]: return list(map(lambda x: x / b, a))

def read_score(record: SaveRecord) -> list[float]: return [
  chapter.values[0]['velocity']
  for chapter in record.history
]

def read_scores(records: list[SaveRecord]) -> list[list[float]]:
  return list(map(read_score, records))

def read_name(record: SaveRecord) -> str:
  representation = record.name.split('-')[-2]
  prefix = record.name.split('/')[1]

  label = ""
  match prefix:
    case "1": return "wlasne-prawd-1 - default"
    case "2": return "wlasne-prawd-2 - all1"
    case "3": return "wlasne-prawd-3 - static"
    case "4": return "wlasne-prawd-4 - dynamic"

  return f"{label}-{representation}"

def groupby_representation(records: list[SaveRecord]): return [
  (representation, list(values))
  for (representation, values) in groupby(records, read_name)
]

def aggregate(records: list[SaveRecord], fn) -> list[float]:
  return list(map(fn, zip(*read_scores(records))))

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
  plt.ylabel("Velocity")
  plt.savefig(f'resources/lab-3/figures/all-boxplot.png', bbox_inches='tight')

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
  plt.ylabel("Velocity")
  plt.legend(
    loc='center left',
    bbox_to_anchor=(0.96, 0.5),
  )
  plt.savefig(f'resources/lab-3/figures/all-plot.png', bbox_inches='tight')

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

  plt.ylabel('Velocity')
  plt.legend(
    loc='center left',
    bbox_to_anchor=(0.96, 0.5),
  )
  plt.savefig(f'resources/lab-3/figures/all-aggregated.png', bbox_inches='tight')


def main():
  records = [resources.read(name, model=SaveRecord) for name in resources.names(f'lab-3')]

  ensure_directory(f'resources/lab-3/figures')
  boxplot(records)
  plot(records)
  aggregated(records)

if __name__ == '__main__':
  main()
