from commands import Command
from commands.utils import invoke

def main():
  commands: list[Command] = []

  def create_command(**kwargs):
    commands.append(
      Command(
        optimization_targets=["vertpos"],
        population=100,
        generations=50,
        max_part_count=50,
        max_joint_count=50,
        max_connection_count=50,
        hall_of_fame_size=1,
        verbose=True,
        **kwargs,
      )
    )

  for genetic_format in ("9",):
    for iteration in range(1, 1 + 1):
      create_command(
        name=f"lab-7/results/test/{genetic_format}/HoF-1-{iteration}",
        sims=[f'eval-allcriteria-mini', "deterministic", 'only-body', f'recording-body-coords-mini'],
        genetic_format=genetic_format,
      )

  invoke(commands)

  print("Processing is done.")

if __name__ == '__main__':
  main()
