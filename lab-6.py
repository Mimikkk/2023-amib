import argparse
import os
import sys
from time import time

from deap import creator, base, tools, algorithms, gp
import numpy as np

from libs.framspy.FramsticksLib import FramsticksLib


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
  BAD_FITNESS = [-1] * len(OPTIMIZATION_CRITERIA)
  genotype = gp.compile(individual, primitives)
  data = lib.evaluate([genotype])
  valid = True
  try:
    first_genotype_data = data[0]
    evaluation_data = first_genotype_data["evaluations"]
    default_evaluation_data = evaluation_data[""]
    if default_evaluation_data['vertpos'] < 0: default_evaluation_data['vertpos'] = 0
    fitness = [default_evaluation_data[crit] for crit in OPTIMIZATION_CRITERIA]

  except (KeyError, TypeError) as e:
    valid = False
    print('Problem "%s" so could not evaluate genotype "%s", hence assigned it low fitness: %s' % (str(e), genotype, BAD_FITNESS))
  if valid:
    default_evaluation_data['numgenocharacters'] = len(genotype)
    valid &= within_constraint(genotype, default_evaluation_data, 'numparts', args.max_numparts)
    valid &= within_constraint(genotype, default_evaluation_data, 'numjoints', args.max_numjoints)
    valid &= within_constraint(genotype, default_evaluation_data, 'numneurons', args.max_numneurons)
    valid &= within_constraint(genotype, default_evaluation_data, 'numconnections', args.max_numconnections)
    valid &= within_constraint(genotype, default_evaluation_data, 'numgenocharacters', args.max_numgenochars)
  if not valid:
    fitness = BAD_FITNESS
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
  creator.create("FitnessMax", base.Fitness, weights=[1.0] * len(OPTIMIZATION_CRITERIA))
  creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

  toolbox = base.Toolbox()
  toolbox.register("expr", gp.genHalfAndHalf, pset=primitives, min_=1, max_=10)
  toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)
  toolbox.register("evaluate", frams_evaluate, lib)
  toolbox.register("mate", gp.cxOnePoint)
  toolbox.register("expr_mut", gp.genFull, min_=1, max_=10)
  toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=primitives)

  if len(OPTIMIZATION_CRITERIA) <= 1:
    toolbox.register("select", tools.selTournament, tournsize=tournament_size)
  else:
    toolbox.register("select", tools.selNSGA2)
  return toolbox
def parse_arguments():
  parser = argparse.ArgumentParser(
    description=f'Run this program with "python -u {sys.argv[0]}" if you want to disable buffering of its output.'
  )
  parser.add_argument(
    '-path',
    type=ensure_dir,
    required=True,
    help='Path to Framsticks CLI without trailing slash.'
  )
  parser.add_argument(
    '-lib',
    required=False,
    help='Library name. If not given, "frams-objects.dll" or "frams-objects.so" is assumed depending on the platform.'
  )
  parser.add_argument(
    '-sim',
    required=False,
    default="eval-allcriteria.sim",
    help="The name of the .sim file with settings for evaluation, mutation, crossover, and similarity estimation. If not given, \"eval-allcriteria.sim\" is assumed by default. Must be compatible with the \"standard-eval\" expdef. If you want to provide more files, separate them with a semicolon ';'."
  )
  parser.add_argument(
    '-genformat',
    required=False,
    help='Genetic format for the simplest initial genotype, for example 4, 9, or B. If not given, f1 is assumed.'
  )
  parser.add_argument(
    '-initialgenotype',
    required=False,
    help='The genotype used to seed the initial population. If given, the -genformat argument is ignored.'
  )
  parser.add_argument(
    '-opt',
    required=True,
    help='optimization criteria: vertpos, velocity, distance, vertvel, lifespan, numjoints, numparts, numneurons, numconnections (or other as long as it is provided by the .sim file and its .expdef). For multiple criteria optimization, separate the names by the comma.'
  )
  parser.add_argument(
    '-popsize',
    type=int,
    default=50,
    help="Population size, default: 50."
  )
  parser.add_argument(
    '-generations',
    type=int,
    default=5,
    help="Number of generations, default: 5."
  )
  parser.add_argument(
    '-tournament',
    type=int,
    default=5,
    help="Tournament size, default: 5."
  )
  parser.add_argument(
    '-pmut',
    type=float,
    default=0.9,
    help="Probability of mutation, default: 0.9"
  )
  parser.add_argument(
    '-pxov',
    type=float,
    default=0.2,
    help="Probability of crossover, default: 0.2"
  )
  parser.add_argument(
    '-hof_size',
    type=int,
    default=10,
    help="Number of genotypes in Hall of Fame. Default: 10."
  )
  parser.add_argument(
    '-hof_savefile',
    required=False,
    help='If set, Hall of Fame will be saved in Framsticks file format (recommended extension *.gen).'
  )
  parser.add_argument(
    '-max_numparts',
    type=int,
    default=None,
    help="Maximum number of Parts. Default: no limit"
  )
  parser.add_argument(
    '-max_numjoints',
    type=int,
    default=None,
    help="Maximum number of Joints. Default: no limit"
  )
  parser.add_argument(
    '-max_numneurons',
    type=int,
    default=None,
    help="Maximum number of Neurons. Default: no limit"
  )
  parser.add_argument(
    '-max_numconnections',
    type=int,
    default=None,
    help="Maximum number of Neural connections. Default: no limit"
  )
  parser.add_argument(
    '-max_numgenochars',
    type=int,
    default=None,
    help="Maximum number of characters in genotype (including the format prefix, if any). Default: no limit"
  )
  return parser.parse_args()
def ensure_dir(string: str):
  if not os.path.isdir(string): raise NotADirectoryError(string)
  return string


def save_genotypes(filename, OPTIMIZATION_CRITERIA, hall_of_fame):
  from libs.framspy.framsfiles import writer as framswriter

  with open(filename, "w") as outfile:
    for individual in hall_of_fame:
      values = {k: individual.fitness.values[i] for i, k in enumerate(OPTIMIZATION_CRITERIA)}

      outfile.write(framswriter.from_collection({"_classname": "org", "genotype": individual[0]} | values))
      outfile.write("\n")

  print(f"Saved '{filename}' ({len(hall_of_fame):d})")


class Tree: pass
class Sequence: pass

def primitive_stick(tail):
  tree = 'X'
  if tail is not None: tree += tail
  return tree

def primitive_parenthesis(tail):
  if not tail: return ''
  if 'X' not in tail: return ''
  return '(' + tail + ')'

def primitive_branch(head, tail):
  tree = ''
  if head is not None: tree += head
  if tail is not None: tree += ',' + tail
  return tree

def primitive_modifier(mod):
  def wrap(tail):
    if not tail: return ''
    return mod + tail
  return wrap

primitives = gp.PrimitiveSetTyped('main', [], Tree)
primitives.addPrimitive(primitive_stick, [Tree], Tree)
primitives.addPrimitive(primitive_parenthesis, [Sequence], Tree)
primitives.addPrimitive(primitive_branch, [Tree, Sequence], Sequence)

for modifier in 'RrQqCcLlWwFf':
  primitives.addPrimitive(primitive_modifier(modifier), [Tree], Tree, name=f'mod_{modifier}')

primitives.addTerminal('X', Tree)
primitives.addTerminal(None, Sequence)
primitives.addTerminal('', Sequence)

if __name__ == "__main__":
  FramsticksLib.DETERMINISTIC = False
  args = parse_arguments()
  print("Argument values:", ", ".join(['%s=%s' % (arg, getattr(args, arg)) for arg in vars(args)]))

  OPTIMIZATION_CRITERIA = args.opt.split(",")
  lib = FramsticksLib(args.path, args.lib, args.sim.split(";"))

  toolbox = prepare_toolbox(
    lib, args.tournament,
    '1' if args.genformat is None else args.genformat,
    args.initialgenotype
  )

  population = toolbox.population(n=args.popsize)
  hof = tools.HallOfFame(args.hof_size)
  stats = tools.Statistics(lambda ind: ind.fitness.values)
  stats.register("avg", np.mean)
  stats.register("stddev", np.std)
  stats.register("min", np.min)
  stats.register("max", np.max)

  t = time()
  population, log = algorithms.eaSimple(
    population,
    toolbox,
    cxpb=args.pxov,
    mutpb=args.pmut,
    ngen=args.generations,
    stats=stats,
    halloffame=hof,
    verbose=True,
    hof_hist_path=f'{args.hof_savefile}_hist.txt'
  )

  t = time() - t
  print('Best individuals:')
  for individual in hof:
    print(individual.fitness, '\t-->\t', gp.compile(individual, primitives))

  if args.hof_savefile is not None:
    path = args.hof_savefile

    save_genotypes(f'{path}_hof.txt', OPTIMIZATION_CRITERIA, hof)
    with open(f'{path}_hof.txt', 'w') as file:
      individual = hof[0]
      print(individual.fitness, file=file)
      print(gp.compile(individual, primitives), file=file)
    with open(f'{path}_log.txt', 'w') as file: print(log, file=file)
    with open(f'{path}_time.txt', 'w') as file: print(t, file=file)
