import asyncio

import numpy as np

from commands import command
import resources
import sims

async def main():
  commands: list[command] = []

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
      command(
        savefile=f"HoF-{name}",
        optimization_targets=["vertpos"],
        population=5,
        generations=1,
        sims=[f'eval-allcriteria.sim', "deterministic.sim", 'sample-period-2.sim', f'{name}.sim'],
        initial_genotype='/*9*/BLU',
        max_part_count=30,
        max_genotype_length=50,
        hall_of_fame_size=1,
      )()
    )

  await asyncio.gather(*commands)

if __name__ == '__main__':
  asyncio.run(main())
