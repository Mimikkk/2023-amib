import argparse
import os
import sys
import numpy as np
from deap import creator, base, tools, algorithms
from FramsticksLib import FramsticksLib
import resources

OptimizationTargets: list[str]
parsed_args: argparse.Namespace
def genotype_within_constraint(genotype, dict_criteria_values, criterion_name, constraint_value):
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
  # fitness of -1 is intended to discourage further propagation of this genotype via selection ("this genotype is very poor")
  BAD_FITNESS = [-1] * len(OptimizationTargets)
  genotype = individual[
    0]  # individual[0] because we can't (?) have a simple str as a deap genotype/individual, only list of str.
  data = frams_lib.evaluate([genotype])
  # print("Evaluated '%s'" % genotype, 'evaluation is:', data)
  valid = True
  try:
    first_genotype_data = data[0]
    evaluation_data = first_genotype_data["evaluations"]
    default_evaluation_data = evaluation_data[""]
    fitness = [default_evaluation_data[crit] for crit in OptimizationTargets]
  except (KeyError,
          TypeError) as e:  # the evaluation may have failed for an invalid genotype (such as X[@][@] with "Don't simulate genotypes with warnings" option) or for some other reason
    valid = False
    print('Problem "%s" so could not evaluate genotype "%s", hence assigned it low fitness: %s' % (
      str(e), genotype, BAD_FITNESS))
  if valid:
    # for consistent constraint checking below
    default_evaluation_data['numgenocharacters'] = len(genotype)
    valid &= genotype_within_constraint(genotype, default_evaluation_data, 'numparts', parsed_args.max_numparts)
    valid &= genotype_within_constraint(genotype, default_evaluation_data, 'numjoints', parsed_args.max_numjoints)
    valid &= genotype_within_constraint(genotype, default_evaluation_data, 'numneurons', parsed_args.max_numneurons)
    valid &= genotype_within_constraint(genotype, default_evaluation_data, 'numconnections',
                                        parsed_args.max_numconnections)
    valid &= genotype_within_constraint(genotype, default_evaluation_data, 'numgenocharacters',
                                        parsed_args.max_numgenochars)
  if not valid:
    fitness = BAD_FITNESS
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


def parse_arguments():
  parser = argparse.ArgumentParser(
    description='Run this program with "python -u %s" if you want to disable buffering of its output.' % sys.argv[0])
  parser.add_argument(
    '-path',
    type=ensure_dir,
    required=True,
    help='Path to Framsticks library without trailing slash.'
  )
  parser.add_argument(
    '-lib',
    required=False,
    help='Library name. If not given, "frams-objects.dll" (or .so or .dylib) is assumed depending on the platform.'
  )
  parser.add_argument(
    '-sim',
    required=False,
    default="eval-allcriteria.sim",
    help="The name of the .sim file with settings for evaluation, mutation, crossover, and similarity estimation. If "
         "not given, \"eval-allcriteria.sim\" is assumed by default. Must be compatible with the \"standard-eval\" "
         "expdef. If you want to provide more files, separate them with a semicolon ';'."
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
    help='optimization criteria: vertpos, velocity, distance, vertvel, lifespan, numjoints, numparts, numneurons, '
         'numconnections (or other as long as it is provided by the .sim file and its .expdef). For multiple criteria '
         'optimization, separate the names by the comma.'
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


def ensure_dir(string):
  if os.path.isdir(string):
    return string
  else:
    raise NotADirectoryError(string)


def save_genotypes(population):
  from framsfiles import writer as framswriter

  # TODO it would be better to save in Individual (after evaluation) all fields returned by Framsticks, and get these fields here, not just the criteria that were actually used as fitness in evolution.
  for individual in population:
    yield (
      framswriter.from_collection(
        {
          "_classname": "org",
          "genotype": individual[0],
        } | {
          criteria: individual.fitness.values[index] for index, criteria in enumerate(OptimizationTargets)
        }
      )
    )


def main():
  global parsed_args, OptimizationTargets

  parsed_args = parse_arguments()

  print(
    "Argument values:",
    ", ".join([f'{argument}={getattr(parsed_args, argument)}' for argument in vars(parsed_args)])
  )

  OptimizationTargets = parsed_args.opt.split(",")

  FramsticksLib.DETERMINISTIC = False
  frams_lib = FramsticksLib(parsed_args.path, parsed_args.lib, parsed_args.sim)

  toolbox = prepare_toolbox(
    frams_lib,
    parsed_args.tournament,
    '1' if parsed_args.genformat is None else parsed_args.genformat,
    parsed_args.initialgenotype
  )

  population = toolbox.population(n=parsed_args.popsize)
  best_population = tools.HallOfFame(parsed_args.hof_size)
  statistics = tools.Statistics(lambda ind: ind.fitness.values)

  statistics.register("avg", np.mean)
  statistics.register("stddev", np.std)
  statistics.register("min", np.min)
  statistics.register("max", np.max)

  algorithms.eaSimple(
    population,
    toolbox,
    cxpb=parsed_args.pxov,
    mutpb=parsed_args.pmut,
    ngen=parsed_args.generations,
    stats=statistics,
    halloffame=best_population,
  )

  print(
    'Best individuals:',
    [f'{individual.fitness}\t-->\t{individual[0]}' for individual in best_population],
    sep='\n'
  )

  if parsed_args.hof_savefile is not None:
    resources.create(f"{parsed_args.hof_savefile}_genotype", '\n'.join(save_genotypes(best_population)))
    resources.create(f"{parsed_args.hof_savefile}_stats", statistics.compile(population))

if __name__ == "__main__":
  main()
