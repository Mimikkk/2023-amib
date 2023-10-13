import asyncio
from dataclasses import dataclass
from typing import Literal, Any, Callable

from constants import constants

OptimizationTarget = Literal[
  'vertpos', 'velocity', 'distance', 'vertvel', 'lifespan', 'numjoints', 'numparts', 'numneurons', 'numconnections'
]
GeneticFormat = Literal['4', '9', 'B', 'f1']

@dataclass
class command(object):
  population: int | None = None
  generations: int | None = None
  sims: list[str] | None = None
  genetic_format: GeneticFormat | None = None
  initial_genotype: str | None = None
  tournament_size: int | None = None
  mutation_probability: float | None = None
  crossover_probability: float | None = None
  hall_of_fame_size: int | None = None
  name: str | None = None
  max_part_count: int | None = None
  max_connection_count: int | None = None
  max_genotype_length: int | None = None
  optimization_targets: list[OptimizationTarget] | None = None

  def __iter__(self):
    def with_optional_flags(initial: list[str], *flags: tuple[str, Any, Callable[Any, str]]):
      for (name, value, *args) in flags:
        if value is None: continue
        converter = str if len(args) == 0 else args[0]
        initial.extend([f'-{name}', converter(value)])
      return initial

    return iter(with_optional_flags(
      [constants.Python, constants.Runfile],
      ("path", constants.Library),
      ("opt", self.optimization_targets, ",".join),
      ("popsize", self.population),
      ("generations", self.generations),
      ("sims", self.sims, ';'.join),
      ("genformat", self.genetic_format),
      ("initialgenotype", self.initial_genotype),
      ("tournament", self.tournament_size),
      ("pmut", self.mutation_probability),
      ("pxov", self.crossover_probability),
      ("hof_size", self.hall_of_fame_size),
      ("hof_savefile", self.name),
      ("max_numparts", self.max_part_count),
      ("max_numconnections", self.max_connection_count),
      ("max_genotype_length", self.max_genotype_length),
    ))

  async def __call__(self):
    process = await asyncio.create_subprocess_exec(
      *self,
      stdout=asyncio.subprocess.PIPE,
      stdin=asyncio.subprocess.PIPE,
    )

    return await process.communicate()
