import numpy as np

from commands import Command
from commands.utils import invoke
import sims

def main():
  commands: list[Command] = []

  for experiment in (1, 2):
    name = f'własne-prawd-{experiment}'
    sims.create(
      name,
      rf"""
      sim_params:
      f0_mut:{f9_mut:.2f}
      """
    )

    commands.append(
      Command(
        name=name,
        optimization_targets=["velocity"],
        population=100,
        generations=50,
        sims=[f'eval-allcriteria', "deterministic", 'sample-period-longest', name],
        initial_genotype='/*9*/BLU',
        max_part_count=15,
        max_joint_count=30,
        max_connection_count=30,
        genetic_format='0',
        max_genotype_length=50,
        hall_of_fame_size=5,
        crossover_probability=0,
        verbose=True
      )
    )

  invoke(commands)
  print("Processing is done.")

if __name__ == '__main__':
  main()
