import argparse
from dataclasses import dataclass
import os
import sys
import numpy as np
from deap import creator, base, tools, algorithms
from FramsticksLib import FramsticksLib
from commands.command import OptimizationTarget
from resources import resources


def ensure_dir(string: str):
  if not os.path.isdir(string): raise NotADirectoryError(string)
  return string

@dataclass
class Arguments(object):
  max_numparts: int
  max_numconnections: int
  max_numjoints: int
  max_numneurons: int
  max_numgenochars: int
  path: str
  lib: str
  sim: str
  genformat: str
  initialgenotype: str
  opt: str
  popsize: int
  generations: int
  tournament: int
  pmut: float
  pxov: float
  hof_size: int
  hof_savefile: str

  @staticmethod
  def parse():
    class parser:
      def __init__(self, *args, **kwargs):
        self.item = argparse.ArgumentParser(*args, **kwargs)

      def add(self, *args, **kwargs):
        self.item.add_argument(*args, **kwargs)
        return self

      def done(self):
        return self.item.parse_args()

    return parser(
      description=f'Run this program with "python -u {sys.argv[0]}" if you want to disable buffering of its output.'
    ).add(
      '-path',
      type=ensure_dir,
      required=True,
      help='Path to Framsticks library without trailing slash.'
    ).add(
      '-opt',
      required=True,
      help='optimization criteria: vertpos, velocity, distance, vertvel, lifespan, numjoints, numparts, numneurons, '
           'numconnections (or other as long as it is provided by the .sim file and its .expdef). For multiple criteria '
           'optimization, separate the names by the comma.'
    ).add(
      '-lib',
      required=False,
      help='Library name. If not given, "frams-objects.dll" (or .so or .dylib) is assumed depending on the platform.'
    ).add(
      '-sim',
      required=False,
      default="eval-allcriteria.sim",
      help="The name of the .sim file with settings for evaluation, mutation, crossover, and similarity estimation. If "
           "not given, \"eval-allcriteria.sim\" is assumed by default. Must be compatible with the \"standard-eval\" "
           "expdef. If you want to provide more files, separate them with a semicolon ';'."
    ).add(
      '-genformat',
      required=False,
      help='Genetic format for the simplest initial genotype, for example 4, 9, or B. If not given, f1 is assumed.'
    ).add(
      '-initialgenotype',
      required=False,
      help='The genotype used to seed the initial population. If given, the -genformat argument is ignored.'
    ).add(
      '-popsize',
      required=False,
      type=int,
      default=50,
      help="Population size, default: 50."
    ).add(
      '-generations',
      required=False,
      type=int,
      default=5,
      help="Number of generations, default: 5."
    ).add(
      '-tournament',
      required=False,
      type=int,
      default=5,
      help="Tournament size, default: 5."
    ).add(
      '-pmut',
      required=False,
      type=float,
      default=0.9,
      help="Probability of mutation, default: 0.9"
    ).add(
      '-pxov',
      required=False,
      type=float,
      default=0.2,
      help="Probability of crossover, default: 0.2"
    ).add(
      '-hof_size',
      required=False,
      type=int,
      default=10,
      help="Number of genotypes in Hall of Fame. Default: 10."
    ).add(
      '-hof_savefile',
      required=False,
      help='If set, Hall of Fame will be saved in Framsticks file format (recommended extension *.gen).'
    ).add(
      '-max_numparts',
      required=False,
      type=int,
      help="Maximum number of Parts. Default: no limit"
    ).add(
      '-max_numjoints',
      required=False,
      type=int,
      help="Maximum number of Joints. Default: no limit"
    ).add(
      '-max_numneurons',
      required=False,
      type=int,
      help="Maximum number of Neural connections. Default: no limit"
    ).add(
      '-max_numconnections',
      required=False,
      type=int,
      help="Maximum number of Neural connections. Default: no limit"
    ).add(
      '-max_numgenochars',
      required=False,
      type=int,
      help="Maximum number of characters in genotype (including the format prefix, if any). Default: no limit"
    ).done()

OptimizationTargets: list[OptimizationTarget]
constants: Arguments

def within_constraint(genotype, dict_criteria_values, criterion_name, constraint_value):
  REPORT_CONSTRAINT_VIOLATIONS = False
  if constraint_value is not None:
    actual_value = dict_criteria_values[criterion_name]
    if actual_value > constraint_value:
      if REPORT_CONSTRAINT_VIOLATIONS:
        print('Genotype "%s" assigned low fitness because it violates constraint "%s": %s exceeds threshold %s' % (
          genotype, criterion_name, actual_value, constraint_value))
      return False
  return True
def frams_evaluate(frams_lib, individual):
  unfit = [-1] * len(OptimizationTargets)
  genotype = individual[0]

  valid = True
  try:
    evaluation = frams_lib.evaluate([genotype])[0]['evaluations'][""]
    fitness = [evaluation[target] for target in OptimizationTargets]

    evaluation['numgenocharacters'] = len(genotype)
    valid &= within_constraint(genotype, evaluation, 'numparts', constants.max_numparts)
    valid &= within_constraint(genotype, evaluation, 'numjoints', constants.max_numjoints)
    valid &= within_constraint(genotype, evaluation, 'numneurons', constants.max_numneurons)
    valid &= within_constraint(genotype, evaluation, 'numconnections', constants.max_numconnections)
    valid &= within_constraint(genotype, evaluation, 'numgenocharacters', constants.max_numgenochars)
    if not valid: return unfit
  except (KeyError, TypeError) as error:
    # the evaluation may have failed for an invalid genotype
    # (such as X[@][@] with "Don't simulate genotypes with warnings" option) or for some other reason.
    print(
      f'Problem "{error}" so could not evaluate genotype "{genotype}", hence assigned it low fitness: {unfit}'
    )
    return unfit

  return fitness
def frams_crossover(frams_lib, individual1, individual2):
  # individual[0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
  geno1 = individual1[0]
  # individual[0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
  geno2 = individual2[0]
  individual1[0] = frams_lib.crossOver(geno1, geno2)
  individual2[0] = frams_lib.crossOver(geno1, geno2)
  return individual1, individual2
def frams_mutate(frams_lib, individual):
  individual[0] = frams_lib.mutate([individual[0]])[
    0]  # individual[0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
  return individual,
def frams_getsimplest(frams_lib, genetic_format, initial_genotype):
  return initial_genotype if initial_genotype is not None else frams_lib.getSimplest(genetic_format)
def prepare_toolbox(frams_lib, tournament_size, genetic_format, initial_genotype):
  creator.create("FitnessMax", base.Fitness, weights=[1.0] * len(OptimizationTargets))
  creator.create("Individual", list, fitness=creator.FitnessMax)

  toolbox = base.Toolbox()
  toolbox.register("attr_simplest_genotype", frams_getsimplest, frams_lib, genetic_format, initial_genotype)
  toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_simplest_genotype, 1)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)
  toolbox.register("evaluate", frams_evaluate, frams_lib)
  toolbox.register("mate", frams_crossover, frams_lib)
  toolbox.register("mutate", frams_mutate, frams_lib)

  if len(OptimizationTargets) <= 1:
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
  else:
    toolbox.register("select", tools.selNSGA2)

  return toolbox

def save_scores(individual): return {
  criteria: individual.fitness.values[index] for (index, criteria) in enumerate(OptimizationTargets)
}
def save_population_scores(population): return [
  save_scores(individual) for individual in population
]
def save_population(population): return [
  {
    "_classname": "org",
    "genotype": individual[0],
    "values": save_scores(individual)
  }
  for individual in population
]

def main():
  global constants, OptimizationTargets
  constants = Arguments.parse()
  OptimizationTargets = constants.opt.split(",")

  FramsticksLib.DETERMINISTIC = False
  frams_lib = FramsticksLib(constants.path, constants.lib, constants.sim)

  toolbox = prepare_toolbox(
    frams_lib,
    constants.tournament,
    '1' if constants.genformat is None else constants.genformat,
    constants.initialgenotype
  )

  population = toolbox.population(n=constants.popsize)
  best_population = tools.HallOfFame(constants.hof_size)
  statistics = tools.Statistics(lambda ind: ind.fitness.values)

  statistics.register("avg", np.mean)
  statistics.register("stddev", np.std)
  statistics.register("min", np.min)
  statistics.register("max", np.max)
  statistics.register("values", lambda _: save_population_scores(best_population))

  _, logbook = algorithms.eaSimple(
    population,
    toolbox,
    cxpb=constants.pxov,
    mutpb=constants.pmut,
    ngen=constants.generations,
    stats=statistics,
    halloffame=best_population,
  )

  if not constants.hof_savefile: return
  resources.create(constants.hof_savefile, {
    "meta": {
      "command": " ".join(sys.argv),
      "arguments": vars(constants),
    },
    "name": constants.hof_savefile,
    "population": save_population(best_population),
    "history": list(logbook)[1:],
  }, format='json')

if __name__ == "__main__":
  main()
