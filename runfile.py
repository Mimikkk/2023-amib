import asyncio
from dataclasses import dataclass
import subprocess
import timeit
from typing import Literal, Any, Callable

import constants
import resources
import serializer
import utils

@dataclass
class command(object):
  population: int | None = None
  generations: int | None = None
  sims: list[str] | None = None
  genetic_format: Literal['4', '9', 'B', 'f1'] | None = None
  initial_genotype: str | None = None
  tournament_size: int | None = None
  mutation_probability: float | None = None
  crossover_probability: float | None = None
  hall_of_fame_size: int | None = None
  name: str | None = None
  max_part_count: int | None = None
  max_connection_count: int | None = None
  max_genotype_length: int | None = None
  optimization_targets: list[str] | None = None

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

async def run_evolution(
    population: int,
    generations: int,
    name: str,

):
  process = await asyncio.create_subprocess_exec(
    *command(
      optimization_targets=["vertpos"],
      population=population,
      generations=generations,
      name=name
    ),
    stdout=asyncio.subprocess.PIPE,
    stdin=asyncio.subprocess.PIPE,
  )

  return await process.communicate()


@utils.timed
async def main(
    name: str,
):
  # await asyncio.gather(run_evolution(40, 1, "a"))
  print(resources.read('a_stats'))
  print(resources.read('a_genotype'))

if __name__ == '__main__':
  asyncio.run(main("named"))
