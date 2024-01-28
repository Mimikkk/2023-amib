from commands import Command
from commands.utils import invoke

def main():
  commands: list[Command] = []

  def create_command(**kwargs):
    commands.append(
      Command(
        optimization_targets=["vertpos"],
        population=100,
        generations=25,
        max_part_count=50,
        max_joint_count=50,
        max_connection_count=50,
        hall_of_fame_size=1,
        verbose=True,
        **kwargs,
      )
    )

  for genetic_format in ("0", "1", "4", "9"):
    for iteration in range(1, 16 + 1):
      create_command(
        name=f"lab-7/results/{genetic_format}/HoF-1-{iteration}",
        sims=[f'eval-allcriteria-mini', "deterministic", 'only-body', f'recording-body-coords-mini'],
        genetic_format=genetic_format,
      )

  for i in range(len(commands) // 16):
    batch = commands[i * 16: (i + 1) * 16]
    invoke(batch)

  print("Processing is done.")

if __name__ == '__main__':
  main()
