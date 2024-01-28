import argparse
from dataclasses import dataclass
import os
from typing import Generator

MODULE_PATH = "./src/__init__.py"
MODULE_NAME = "src"
import importlib.util as utl
import sys
spec = utl.spec_from_file_location(MODULE_NAME, MODULE_PATH)
module = utl.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

from deap import creator, base, tools, algorithms
import numpy as np

from FramsticksLib import FramsticksLib
import src.libs.framspy.frams as frams
from src.commands.command import OptimizationTarget
from src.resources import resources


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
  opt: list[str]
  popsize: int
  generations: int
  tournament: int
  pmut: float
  pxov: float
  hof_size: int
  hof_savefile: str
  parameter_scheduler_parameters: dict
  parameter_scheduler_factor: dict


  @staticmethod
  def parse() -> 'Arguments':
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
      type=lambda string: string.split(','),
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
      type=str,
      default='1',
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
    ).add(
      '-parameter_scheduler_factor',
      required=False,
      default=1.0,
      type=float,
      help="Factor by which the parameters are multiplied after each generation. Default: 1.0"
    ).add(
      '-parameter_scheduler_parameters',
      required=False,
      default="",
      type=lambda arg: {key: 'noop' for key in arg.split(",")} if arg else {},
      help="Parameters to be used by the scheduler. Expected format: parameter=initial_value;parameter< =initial_value >."
           "Default: no parameters"
    ).done()

OptimizationTargets: list[OptimizationTarget]
constants: Arguments
ParameterScheduler: Generator[dict[str, float], None, None]

def monkey_patch_genman():
  frams.GenMan.__dict__['get'] = lambda key: frams.GenMan.__getattr__(key)._value()
  frams.GenMan.__dict__['set'] = frams.GenMan.__setattr__

def create_parameters_scheduler(params: dict[str, float], factor: float):
  while True:
    for (key, value) in params.items():
      params[key] *= factor

    yield params

def within_constraint(genotype, values, criterion, max_value):
  REPORT_CONSTRAINT_VIOLATIONS = False
  if max_value is None: return True
  actual_value = values[criterion]

  if actual_value > max_value:
    if REPORT_CONSTRAINT_VIOLATIONS:
      print(
        f'Genotype "{genotype}" assigned low fitness because it violates constraint "{criterion}": {actual_value} exceeds threshold {max_value}'
      )
    return False
  return True

def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
def convex_hull(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
  points = sorted(set(points))
  if len(points) < 3: return points

  lower = []
  for point in points:
    while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0: lower.pop()
    lower.append(point)

  upper = []
  for point in reversed(points):
    while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0: upper.pop()
    upper.append(point)
  return lower[:-1] + upper[:-1]


def shoelace_area(parts) -> float:
  return 0.5 * abs(sum(x0 * y1 - x1 * y0 for ((x0, y0, _), (x1, y1, _)) in zip(parts, parts[1:] + parts[:1])))

def around_zero(value, epsilon=1e-6): return abs(value) < epsilon

def frams_evaluate(lib, individual):
  unfit = [-1] * len(OptimizationTargets)
  genotype = individual[0]

  valid = True
  try:
    evaluation = lib.evaluate([genotype])[0]['evaluations'][""]
    positions: list[tuple[float, float, float]] = [(x, y, z) for (x, y, z) in evaluation["data->bodyrecording"]]

    area = shoelace_area(convex_hull(positions))
    fitness = [-10 if around_zero(area) else area]

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

def frams_update_sim_params():
  params = next(ParameterScheduler)

  for (key, value) in params.items():
    frams.GenMan[key] = value

def frams_crossover(lib, first, second):
  # individual[0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
  geno1 = first[0]
  # individual[0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
  geno2 = second[0]
  first[0] = lib.crossOver(geno1, geno2)
  second[0] = lib.crossOver(geno1, geno2)
  return first, second
def frams_mutate(lib, individual):
  individual[0] = lib.mutate([individual[0]])[0]
  return individual,

def frams_getsimplest(lib, genetic_format, initial_genotype):
  return initial_genotype if initial_genotype else lib.getSimplest(genetic_format)
def prepare_toolbox(lib, tournament_size, genetic_format, initial_genotype):
  creator.create("FitnessMax", base.Fitness, weights=[1.0] * len(OptimizationTargets))
  creator.create("Individual", list, fitness=creator.FitnessMax)

  toolbox = base.Toolbox()
  toolbox.register("update_sim_params", frams_update_sim_params)
  toolbox.register("attr_simplest_genotype", frams_getsimplest, lib, genetic_format, initial_genotype)
  toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_simplest_genotype, 1)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)
  toolbox.register("evaluate", frams_evaluate, lib)
  toolbox.register("mate", frams_crossover, lib)
  toolbox.register("mutate", frams_mutate, lib)

  if len(OptimizationTargets) <= 1:
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
  else:
    toolbox.register("select", tools.selNSGA2)

  return toolbox

def save_scores(individual): return {
  'fitness': individual.fitness.values[index] for (index, criteria) in enumerate(OptimizationTargets)
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
  global constants, OptimizationTargets, ParameterScheduler
  constants = Arguments.parse()
  frams_lib = FramsticksLib(constants.path, constants.lib, constants.sim)
  monkey_patch_genman()

  constants.parameter_scheduler_parameters = {
    key: frams.GenMan[key]
    for key in constants.parameter_scheduler_parameters
  }

  OptimizationTargets = constants.opt
  ParameterScheduler = create_parameters_scheduler(constants.parameter_scheduler_parameters, constants.parameter_scheduler_factor)
  FramsticksLib.DETERMINISTIC = False

  toolbox = prepare_toolbox(
    frams_lib,
    constants.tournament,
    constants.genformat,
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
    "meta": {"command": " ".join(sys.argv), "arguments": vars(constants)},
    "name": constants.hof_savefile,
    "population": save_population(best_population),
    "history": list(logbook)[1:],
  }, format='json')

if __name__ == "__main__":
  main()
