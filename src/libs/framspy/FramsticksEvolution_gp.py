import argparse
from dataclasses import dataclass
import os

MODULE_PATH = "./src/__init__.py"
MODULE_NAME = "src"
import importlib.util as utl
import sys
spec = utl.spec_from_file_location(MODULE_NAME, MODULE_PATH)
module = utl.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

from deap import creator, base, tools, algorithms, gp
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

def monkey_patch_genman():
  frams.GenMan.__dict__['get'] = lambda key: frams.GenMan.__getattr__(key)._value()
  frams.GenMan.__dict__['set'] = frams.GenMan.__setattr__
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
def frams_evaluate(lib: FramsticksLib, individual):
  unfit = [-1] * len(OptimizationTargets)
  genotype = gp.compile(individual, primitives)
  data = lib.evaluate([genotype])
  valid = True
  try:
    first_genotype_data = data[0]
    values = first_genotype_data["evaluations"][""]
    fitness = [values[target] for target in OptimizationTargets]

  except (KeyError, TypeError) as e:
    valid = False
    print('Problem "%s" so could not evaluate genotype "%s", hence assigned it low fitness: %s' % (str(e), genotype, unfit))
  if valid:
    values['numgenocharacters'] = len(genotype)
    valid &= within_constraint(genotype, values, 'numparts', constants.max_numparts)
    valid &= within_constraint(genotype, values, 'numjoints', constants.max_numjoints)
    valid &= within_constraint(genotype, values, 'numneurons', constants.max_numneurons)
    valid &= within_constraint(genotype, values, 'numconnections', constants.max_numconnections)
    valid &= within_constraint(genotype, values, 'numgenocharacters', constants.max_numgenochars)
  if not valid: return unfit
  return fitness

def frams_crossover(lib: FramsticksLib, first, second):
  gen1 = first[0]
  gen2 = second[0]
  first[0] = lib.crossOver(gen1, gen2)
  second[0] = lib.crossOver(gen1, gen2)
  return first, second
def frams_mutate(lib: FramsticksLib, individual):
  individual[0] = lib.mutate([individual[0]])[0]
  return individual,
def frams_getsimplest(lib: FramsticksLib, genetic_format, initial_genotype):
  return initial_genotype if initial_genotype is not None else lib.getSimplest(genetic_format)
def prepare_toolbox(lib: FramsticksLib, tournament_size, genetic_format, initial_genotype):
  creator.create("FitnessMax", base.Fitness, weights=[1.0] * len(OptimizationTargets))
  creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

  # toolbox = base.Toolbox()
  # toolbox.register("attr_simplest_genotype", frams_getsimplest, lib, genetic_format, initial_genotype)
  # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_simplest_genotype, 1)
  # toolbox.register("population", tools.initRepeat, list, toolbox.individual)
  # toolbox.register("evaluate", frams_evaluate, lib)
  # toolbox.register("mate", frams_crossover, lib)
  # toolbox.register("mutate", frams_mutate, lib)
  toolbox = base.Toolbox()
  toolbox.register("expr", gp.genHalfAndHalf, pset=primitives, min_=1, max_=10)
  toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)
  toolbox.register("evaluate", frams_evaluate, lib)
  toolbox.register("mate", gp.cxOnePoint)
  toolbox.register("expr_mut", gp.genFull, min_=1, max_=10)
  toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=primitives)

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
    "genotype": gp.compile(individual, primitives),
    "values": save_scores(individual)
  }
  for individual in population
]


class Node: pass
class Tree: pass

def primitive_node(tail):
  if not tail: return 'X'
  return f"X{tail}"

def primitive_parenthesis(tail):
  if not tail or 'X' not in tail: return ''
  return f'({tail})'

def primitive_split(head, tail):
  return f'{head if head else ""}{f",{tail}" if tail else ""}'

def create_primitive_modifier(modifier):
  def primitive(tail):
    if not tail: return ''
    return f"{modifier}{tail}"
  return primitive

primitives = gp.PrimitiveSetTyped('main', [], Node)
primitives.addPrimitive(primitive_node, [Node], Node)
primitives.addPrimitive(primitive_parenthesis, [Tree], Node)
primitives.addPrimitive(primitive_split, [Node, Tree], Tree)
modifiers = ['R', 'r', 'Q', 'q', 'C', 'c', 'L', 'l', 'W', 'w', 'F', 'f']
for modifier in modifiers:
  primitives.addPrimitive(create_primitive_modifier(modifier), [Node], Node, name=f'modifier_{modifier}')

primitives.addTerminal('X', Node)
primitives.addTerminal(None, Tree)
primitives.addTerminal('', Tree)

def main():
  global constants, OptimizationTargets
  constants = Arguments.parse()
  lib = FramsticksLib(constants.path, constants.lib, constants.sim)
  monkey_patch_genman()

  constants.parameter_scheduler_parameters = {
    key: frams.GenMan[key]
    for key in constants.parameter_scheduler_parameters
  }

  OptimizationTargets = constants.opt
  FramsticksLib.DETERMINISTIC = False

  toolbox = prepare_toolbox(
    lib,
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
