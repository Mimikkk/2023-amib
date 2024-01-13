from collections import defaultdict
import itertools
from itertools import groupby
from typing import Iterable, Generator

import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

from figures import ensure_directory
import src.resources as resources
from src.resources.models import SaveRecord


def read_name(record: SaveRecord) -> str:
  return record.meta.arguments['genformat']

def groupby_representation(records: list[SaveRecord]) -> dict[str, list[SaveRecord]]: return {
  representation: list(values)
  for (representation, values) in groupby(records, read_name)
}

# def aggregated(records: list[SaveRecord]):
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
#     plt.plot(ticks, averages, label=f"f{representation}")
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
    if score < 0: continue
    mutations.append(mutation)
    scores.append(score)

  return mutations, scores

def generate_crossovers(lib, first, second, count: int):
  crossovers = []
  scores = []
  iters = 0
  while len(crossovers) < count:
    iters += 1
    if iters > 100: break
    cross = lib.crossOver(first, second)
    if cross in crossovers: continue
    [score] = frams_evaluate(lib, [cross])
    if score < 0: continue
    crossovers.append(cross)
    scores.append(score)

  return crossovers, scores

import os
import pickle
from functools import wraps

Individual = tuple[str, float]
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
def create_mutations(records: list[SaveRecord]) -> dict[
  str, list[tuple[tuple[str, float], tuple[list[str], list[float]]]]
]:
  from libs.framspy.FramsticksLib import FramsticksLib
  lib = FramsticksLib(*map(records[0].meta.arguments.get, ('path', 'lib', 'sim')))

  processed = {}
  groups = groupby_representation(records)
  for (representation, group) in groups.items():
    items = []
    for (i, record) in enumerate(group):
      print(f'creating mutations for {representation} | {i + 1}/{len(group)}')
      genotype = record.population[0].genotype
      score = record.population[0].values["vertpos"]
      items.append(((genotype, score), generate_mutations(lib, [genotype], 32)))
    processed[representation] = items
  return processed

@cache_pickle('.lab-4-pickle-crossovers', 'crossovers')
def create_crossovers(records: list[SaveRecord]) -> dict[
  str, list[tuple[tuple[Individual, Individual], tuple[list[str], list[float]]]]
]:
  from libs.framspy.FramsticksLib import FramsticksLib
  lib = FramsticksLib(*map(records[0].meta.arguments.get, ('path', 'lib', 'sim')))

  processed = {}
  groups = groupby_representation(records)
  for (representation, group) in groups.items():
    group = group[:100]
    items: list[tuple[tuple[tuple[str, float], tuple[str, float]], tuple[list[str], list[float]]]] = []
    combination_count = len(group) * (len(group) - 1) // 2
    for (i, (record_a, record_b)) in enumerate(itertools.combinations(group, r=2)):
      print(f'creating crossovers for {representation} | {i + 1}/{combination_count}')
      genotype_a = record_a.population[0].genotype
      score_a = record_a.population[0].values["vertpos"]
      genotype_b = record_b.population[0].genotype
      score_b = record_b.population[0].values["vertpos"]
      crossovers = generate_crossovers(lib, genotype_a, genotype_b, 2)
      if not crossovers: continue
      items.append((((genotype_a, score_a), (genotype_b, score_b)), crossovers))
    processed[representation] = items
  return processed

@cache_pickle('.lab-4-pickle-random-walk', 'random-walk')
def random_walks(records: Iterable[SaveRecord]):
  from libs.framspy.FramsticksLib import FramsticksLib
  lib = FramsticksLib(*map(records[0].meta.arguments.get, ('path', 'lib', 'sim')))

  def create_population(records: list[SaveRecord]) -> list[Individual]: return [
    (record.population[0].genotype, record.population[0].values["vertpos"])
    for record in records
  ]
  def create_buckets(population: Iterable[Individual], count: int):
    min_score = min(score for _, score in population)
    max_score = max(score for _, score in population)
    print(min_score, max_score)
    buckets = {
      (a, b): []
      for (a, b) in itertools.pairwise(np.linspace(min_score, max_score + 0.05, count + 1))
    }

    for (genotype, score) in population:
      for (lower, upper), bucket in buckets.items():
        if lower <= score < upper:
          bucket.append(genotype)
          break

    return buckets
  def random_walk(start: str, iterations: int) -> Generator[Individual, None, None]:
    origin = [start]
    [score] = frams_evaluate(lib, origin)
    yield origin, score

    while iterations > 0:
      mutated = lib.mutate(origin)
      [score] = frams_evaluate(lib, mutated)
      if score < 0: continue
      iterations -= 1
      origin = mutated
      yield origin, score

  processed = {}
  for (representation, group) in groupby_representation(records).items():
    population = create_population(group)
    buckets = create_buckets(population, 5)

    items = defaultdict(list)
    for (lower, upper), bucket in buckets.items():
      for (i, genotype) in enumerate(bucket):
        print(f'creating random walks for {representation} | {lower:.2f} - {upper:.2f} | {i + 1}/{len(bucket)}')

        walk = tuple(random_walk(genotype, 25))
        items[(lower, upper)].append(walk)

    processed[representation] = items
  return processed

def main():
  records = read_records()

  for (representation, items) in groupby_representation(records).items():
    print(f"before {representation} prunning: {len(items):>5}")

  records = prune_records(records, 0.05)
  for (representation, items) in groupby_representation(records).items():
    print(f"after  {representation} prunning: {len(items):>5}")

  # boxplot(records)
  # aggregated(records)

  mutations = create_mutations(records)
  crossovers = create_crossovers(records)
  walks = random_walks(records)

  ensure_directory(f'resources/lab-4/figures/1')
  def scatterplots(mutations: dict[str, list[tuple[str, list[float]]]]):
    for (representation, group) in mutations.items():
      points_x = []
      points_y = []
      for ((original, score), (mutations, scores)) in group:
        y = [score] + scores
        x = [score] * len(y)
        points_x.extend(x)
        points_y.extend(y)

      plt.tight_layout()
      plt.figure(figsize=(12, 5))
      plt.scatter(points_x, points_y, alpha=0.1, color='green')
      plt.title(f"Representation f{representation}")
      plt.ylabel("Mutation Fitness")
      plt.xlabel("Original Fitness")
      plt.savefig(f'resources/lab-4/figures/1/mutations-{representation}.png', bbox_inches='tight')
      # plt.show()
  def heatmaps(crossovers):
    plt.tight_layout()
    plt.figure(figsize=(12, 5))

    for (representation, group) in crossovers.items():
      parent_fitness_of = (
          {a: score for ((a, score), _), _ in group}
          |
          {b: score for (_, (b, score)), _ in group}
      )
      child_fitness_of = (
        {(a, b): score for ((a, _), (b, _)), (_, [score, *_]) in group}
      )
      parents_a = sorted([a for ((a, _), _), _ in group], key=lambda a: parent_fitness_of[a])
      parents_b = sorted([b for (_, (b, _)), _ in group], key=lambda b: parent_fitness_of[b])

      points = np.zeros((len(parents_b), len(parents_a)))
      n = len(parents_a)
      for (i, b) in enumerate(parents_b):
        for (j, a) in enumerate(parents_a):
          points[n - i - 1, j] = child_fitness_of.get((a, b)) or 0
      sns.heatmap(points)
      plt.title(f"Representation f{representation}")
      plt.xlabel("First Parent Fitness")
      plt.ylabel("Second Parent Fitness")
      plt.xticks([0, (len(points) - 1) // 2, len(points) - 1], ['0', '1', f'{parent_fitness_of[parents_b[-1]]:.2f}'], rotation=0)
      plt.yticks([0, (len(points) - 1) // 2, len(points) - 1], [f"{parent_fitness_of[parents_a[-1]]:.2f}", '1', '0'], rotation=0)
      plt.savefig(f'resources/lab-4/figures/1/crossovers-{representation}.png', bbox_inches='tight')
      plt.show()
  def walkplots(walks):
    for (representation, buckets) in walks.items():
      plt.tight_layout()
      plt.figure(figsize=(12, 5))
      legends = [f'Range: [{a:0.2f}-{b:0.2f})' for (a, b) in buckets]
      legends[-1] = legends[-1].replace(')', ']')
      plt.title(f'Representation f{representation}')
      plt.xlabel('Iteration')
      plt.ylabel('Fitness')

      ticks = range(0, 26)
      buckets_scores = {ranges: [[score for (_, score) in run] for run in runs] for (ranges, runs) in buckets.items()}

      for i, (ranges, runs) in enumerate(buckets_scores.items()):
        # print(len(runs))

        averages = [np.mean([run[iteration] for run in runs]) for iteration in ticks]
        stddevs = [np.std([run[iteration] for run in runs]) for iteration in ticks]
        stddevs = [stddev / 5 for stddev in stddevs]
        print(averages, stddevs)


        def addlists(a, b): return [x + y for (x, y) in zip(a, b)]
        def sublists(a, b): return [x - y for (x, y) in zip(a, b)]

        plt.plot(ticks, averages, label=legends[i])
        plt.fill_between(
          ticks,
          addlists(averages, stddevs),
          sublists(averages, stddevs),
          alpha=0.3
        )

        # plt.savefig(f'resources/lab-4/figures/1/walks-{representation}.png', bbox_inches='tight')
      plt.legend(
        loc='center left',
        bbox_to_anchor=(0.96, 0.5),
      )
      plt.savefig(f'resources/lab-4/figures/1/walks-{representation}.png', bbox_inches='tight')
      plt.show()

  # scatterplots(mutations)
  # heatmaps(crossovers)
  walkplots(walks)


if __name__ == '__main__':
  main()
