import asyncio
from dataclasses import dataclass
import subprocess
from typing import Literal, Any

import constants
import resources

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
  hall_of_fame_path: str | None = None
  max_part_count: int | None = None
  max_connection_count: int | None = None
  max_genotype_length: int | None = None
  opt: str | None = None

  def as_arguments(self):
    def with_optional_flags(initial: list[str], *flags: tuple[str, Any]):
      for (name, value) in flags:
        if value is None: continue
        initial.extend([f'-{name}', str(value)])
      return initial

    return with_optional_flags(
      [constants.Python, constants.Runfile],
      ("path", constants.Library),
      ("opt", self.opt),
      ("popsize", self.population),
      ("generations", self.generations),
      ("sims", self.sims),
      ("genformat", self.genetic_format),
      ("initialgenotype", self.initial_genotype),
      ("tournament", self.tournament_size),
      ("pmut", self.mutation_probability),
      ("pxov", self.crossover_probability),
      ("hof_size", self.hall_of_fame_size),
      ("hof_savefile", self.hall_of_fame_path),
      ("max_numparts", self.max_part_count),
      ("max_numconnections", self.max_connection_count),
      ("max_genotype_length", self.max_genotype_length),
    )

  def __iter__(self):
    return iter(self.as_arguments)

async def run_evolution(
    population: int,
    generations: int,
    hall_of_fame_path: str,

):
  process = await asyncio.create_subprocess_exec(
    *command(
      opt="vertpos",
      population=population,
      generations=generations,
      hall_of_fame_path=resources.pathof(hall_of_fame_path)
    ),
    stdout=asyncio.subprocess.PIPE,
    stdin=asyncio.subprocess.PIPE,
  )
  return await process.communicate()


async def main(
    name: str,
):
  print(await run_evolution(40, 1, name))

if __name__ == '__main__':
  asyncio.run(main("named"))
