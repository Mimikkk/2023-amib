from commands import Command
from commands.utils import invoke

def main():
  commands: list[Command] = []

  def create_command(**kwargs):
    commands.append(
      Command(
        optimization_targets=["vertpos"],
        population=100,
        generations=70,
        max_part_count=30,
        max_joint_count=30,
        max_neuron_count=20,
        max_connection_count=30,
        hall_of_fame_size=1,
        verbose=True,
        **kwargs,
      )
    )

  for genetic_format in ("1",):
    for iteration in range(1, 64 + 1):
      create_command(
        name=f"lab-5/results/1/HoF-vert-{genetic_format}-{iteration}",
        sims=[f'eval-allcriteria', "deterministic", 'sample-period-longest'],
        genetic_format=genetic_format,
      )

  batches = len(commands) // 16
  for i in range(batches):
    batch = commands[i * 16: (i + 1) * 16]
    invoke(batch)

  print("Processing is done.")

if __name__ == '__main__':
  main()
