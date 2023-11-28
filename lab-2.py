from commands import Command
from commands.utils import invoke

prefix = '4'

def main():
  commands: list[Command] = []

  for genetic_format in ("0", "1", "4", "9"):
    for iteration in range(1, 11):
      name = f'lab-2/{prefix}/results/HoF-f{genetic_format}-{iteration}'

      commands.append(
        Command(
          name=name,
          sims=[f'eval-allcriteria', "deterministic", 'sample-period-2', 'only-body'],
          optimization_targets=["vertpos", "numparts"],
          population=100,
          max_part_count=30,
          genetic_format=genetic_format,
          generations=50,
          hall_of_fame_size=1,
        )
      )

  invoke(commands)
  print("Processing is done.")

if __name__ == '__main__':
  main()
