from commands import Command
from commands.utils import invoke

prefix = '2'

def main():
  commands: list[Command] = []

  for genetic_format in ("0", "1", "4", "9"):
    name = f'HoF-{prefix}-f{genetic_format}'

    commands.append(
      Command(
        name=name,
        sims=[f'eval-allcriteria', "deterministic", 'sample-period-2', 'only-body'],
        optimization_targets=["vertpos"],
        population=100,
        max_part_count=30,
        genetic_format=genetic_format,
        generations=50,
        hall_of_fame_size=5,
      )
    )

  invoke(commands)
  print("Processing is done.")

if __name__ == '__main__':
  main()
