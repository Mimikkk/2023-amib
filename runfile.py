from dataclasses import dataclass
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

  @property
  def flags(self):
    def with_optional_flags(initial: list[str], *conditional: tuple[str, Any]):
      initial.extend(f"-{name} {value}" for (name, value) in conditional if value is not None)
      return initial

    return with_optional_flags(
      [constants.Python, constants.Runfile],
      ("path", constants.Library),
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

def run_evolution(
    population: int,
    generations: int,
    hall_of_fame_path: str,

):
  print((command(population=population, generations=generations,
                 hall_of_fame_path=resources.pathof(hall_of_fame_path)).flags))

def main(
    name: str,
):
  run_evolution(40, 30, name)
  print(resources.listed())

if __name__ == '__main__':
  main("test")
