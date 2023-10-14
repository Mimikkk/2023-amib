import numpy as np

from commands import Command
import sims

def main():
  commands: list[Command] = []

  for f9_mut in np.linspace(0, 0.5, 10 + 1):
    name = f'f9-mut-{f9_mut:.2f}'
    sims.create(
      name,
      rf"""
      sim_params:
      f9_mut:{f9_mut:.2f}                  
      """
    )

    commands.append(
      Command(
        name=f"HoF-{name}",
        optimization_targets=["vertpos"],
        population=5,
        generations=1,
        sims=[f'eval-allcriteria', "deterministic", 'sample-period-2', name],
        initial_genotype='/*9*/BLU',
        max_part_count=30,
        max_genotype_length=50,
        hall_of_fame_size=5,
        verbose=True
      )
    )

  processes = [command() for command in commands]

  for process in processes: process.join()
  print("Processing is done.")

if __name__ == '__main__':
  main()
