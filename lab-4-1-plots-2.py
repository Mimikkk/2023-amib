from itertools import groupby
from typing import TypeVar, Callable

from matplotlib import pyplot as plt
import numpy as np

from figures import ensure_directory
from libs.framspy.FramsticksLib import FramsticksLib
import src.resources as resources
from src.resources.models import SaveRecord


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

def groupby_representation(records: list[SaveRecord]) -> dict[str, list[SaveRecord]]: return {
  representation: list(values)
  for (representation, values) in groupby(records, read_name)
}

T = TypeVar('T')
def aggregate(records: list[SaveRecord], fn: Callable[[SaveRecord], T]) -> list[T]:
  return list(map(fn, zip(*read_scores(records))))

def boxplot(first: list[SaveRecord]):
  plt.tight_layout()
  plt.figure(figsize=(12, 5))

  representations = {
    representation: aggregate(values, np.mean)
    for representation, values in groupby_representation(first).items()
  }

  plt.boxplot(representations.values(), labels=representations.keys())
  plt.xticks(rotation=15)
  plt.xlabel("Representation")
  plt.ylabel("Vertpos")
  plt.savefig(f'resources/lab-4/figures/1/all-boxplot.png', bbox_inches='tight')

def aggregated(records: list[SaveRecord]):
  plt.tight_layout()
  plt.figure(figsize=(12, 5))

  generations = len(records[0].history)
  representations = {
    representation: (aggregate(values, np.mean), aggregate(values, np.std))
    for representation, values in groupby_representation(records).items()
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

  plt.ylabel('Vertpos')
  plt.legend(
    loc='center left',
    bbox_to_anchor=(0.96, 0.5),
  )
  plt.savefig(f'resources/lab-4/figures/1/all-aggregated.png', bbox_inches='tight')

def within_constraint(genotype, values, criterion, max_value):
  REPORT_CONSTRAINT_VIOLATIONS = True
  if max_value is None: return True
  actual_value = values[criterion]

  if actual_value > max_value:
    if REPORT_CONSTRAINT_VIOLATIONS:
      print(
        f'Genotype "{genotype}" assigned low fitness because it violates constraint "{criterion}": {actual_value} exceeds threshold {max_value}'
      )
    return False
  return True

def frams_evaluate(lib, individual):
  unfit = [-1] * 1
  genotype = individual[0]

  valid = True
  try:
    evaluation = lib.evaluate([genotype])[0]['evaluations'][""]
    fitness = [evaluation["vertpos"]]

    # fitness = [
    #   evaluation[target]
    #   if evaluation[target] > 0 else
    #   evaluation[target] + (evaluation['numparts'] / constants.max_numparts / 5)
    #   for target in OptimizationTargets
    # ]

    evaluation['numgenocharacters'] = len(genotype)
    valid &= within_constraint(genotype, evaluation, 'numparts', 15)
    valid &= within_constraint(genotype, evaluation, 'numjoints', 30)
    valid &= within_constraint(genotype, evaluation, 'numneurons', 20)
    valid &= within_constraint(genotype, evaluation, 'numconnections', 30)
    valid &= within_constraint(genotype, evaluation, 'numgenocharacters', None)
    if not valid: return unfit
  except (KeyError, TypeError) as error:
    # the evaluation may have failed for an invalid genotype
    # (such as X[@][@] with "Don't simulate genotypes with warnings" option) or for some other reason.
    print(
      f'Problem "{error}" so could not evaluate genotype "{genotype}", hence assigned it low fitness: {unfit}'
    )
    return unfit

  return fitness

def generate_mutations(lib, individual, count: int):
  mutations = []
  scores = []
  while len(mutations) < count:
    mutated = lib.mutate(individual)
    [mutation] = mutated
    if mutation in mutations: continue
    [score] = frams_evaluate(lib, mutated)
    if score == -1: continue
    mutations.append(mutation)
    scores.append(score)

  return mutations, scores

import os
import pickle
from functools import wraps

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

@cache_pickle('.lab-4-pickle-records', 'records')
def read_records():
  return [resources.read(name, model=SaveRecord) for name in resources.names("./lab-4/results")]

def prune_records(records: list[SaveRecord], threshold: float):
  return [record for record in records if record.population[0].values["vertpos"] > threshold]

@cache_pickle('.lab-4-pickle-mutations', 'mutations')
def create_mutations(lib: FramsticksLib, records: list[SaveRecord]):
  processed = {}
  groups = groupby_representation(records)
  for (representation, group) in groups.items():
    items = []
    for (i, record) in enumerate(group):
      print(f'creating mutations for {representation} | {i + 1}/{len(group)}')
      genotype = record.population[0].genotype
      items.append((genotype, generate_mutations(lib, [genotype], 32)))
    processed[representation] = items
  return processed

def main():
  records = read_records()
  for (representation, items) in groupby_representation(records).items():
    print(f"before {representation} prunning: {len(items):>5}")

  records = prune_records(records, 0.05)
  for (representation, items) in groupby_representation(records).items():
    print(f"after  {representation} prunning: {len(items):>5}")

  lib = FramsticksLib(*map(records[0].meta.arguments.get, ('path', 'lib', 'sim')))
  mutations = create_mutations(lib, records)

  ensure_directory(f'resources/lab-4/figures/1')
  print(mutations['9'][0][0])

  for (original, (mutations, scores)) in mutations['4']:
    score = frams_evaluate(lib, [original])
    print(score, original, mutations, scores)

    y = [score[0]] + scores
    x = [original] * len(y)
    print(len(x), len(y))
    plt.tight_layout()
    plt.figure(figsize=(12, 5))

    plt.scatter(x, y)
    # plt.xlabel("Original")
    # plt.ylabel("Mutation")
    plt.show()
    break

  # def scatter_plot(datapoints: list[tuple[float, float]]):
  #   plt.tight_layout()
  #   plt.figure(figsize=(12, 5))
  #
  #   generations = len(records[0].history)
  #   representations = {
  #     representation: (aggregate(values, np.mean), aggregate(values, np.std))
  #     for representation, values in groupby_representation(records).items()
  #   }
  #
  #   ticks = range(1, generations + 1)
  #   for (representation, (averages, stddevs)) in representations.items():
  #     plt.plot(ticks, averages, label=representation)
  #     stddevs = [
  #       stddev / 5
  #       for stddev in stddevs
  #     ]
  #
  #     plt.fill_between(
  #       ticks,
  #       dif(averages, stddevs),
  #       sum(averages, stddevs),
  #       alpha=0.3
  #     )
  #
  #   plt.ylabel('Vertpos')
  #   plt.legend(
  #     loc='center left',
  #     bbox_to_anchor=(0.96, 0.5),
  #   )
  #   plt.savefig(f'resources/lab-4/figures/1/all-aggregated.png', bbox_inches='tight')


  # create point plot x fitness, y - mutation fitness


  # boxplot(records)
  # aggregated(records)

if __name__ == '__main__':
  main()
