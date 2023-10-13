import asyncio
from commands import command
import resources

async def main(name: str):
  await command(
    optimization_targets=["vertpos"],
    name=name,
    population=40,
    generations=1
  )()
  print(resources.read(f'{name}_stats'))
  print(resources.read(f'{name}_genotype'))

if __name__ == '__main__':
  asyncio.run(main("named"))
