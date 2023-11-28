from matplotlib import pyplot as plt
from matplotlib.pyplot import colormaps
import numpy as np

from figures import ensure_directory
import resources
from resources.models import SaveRecord

def boxplot(records: list[SaveRecord]):
  plt.tight_layout()
  plt.figure(figsize=(12, 5))

  names = [entry.name for entry in records]
  scores = [
    [individual.values["vertpos"] for individual in entry.population]
    for entry in records
  ]

  plt.boxplot(scores, labels=names)
  plt.xticks(rotation=15)
  plt.xlabel("Fitness")
  plt.ylabel("Representation")
  plt.savefig('resources/lab-1/figures/boxplot.png', bbox_inches='tight')

def plot(records: list[SaveRecord]):
  plt.tight_layout()
  plt.figure(figsize=(12, 5))

  generations = len(records[0].history)
  ticks = range(1, generations + 1)

  cmap = colormaps['tab20c']
  for (index, record) in enumerate(records):
    individuals_scores = list(zip(*[
      [values["vertpos"] for values in generation.values]
      for generation in record.history
    ]))

    for (i, individual_scores) in enumerate(individuals_scores):
      if i == 0:
        plt.plot(ticks, individual_scores, label=record.name, color=cmap(index))
      else:
        plt.plot(ticks, individual_scores, color=cmap(index))

  plt.xlabel("Generation")
  plt.ylabel("Fitness")
  plt.legend(
    loc='center left',
    bbox_to_anchor=(0.96, 0.5),
  )
  plt.savefig('resources/lab-1/figures/plot.png', bbox_inches='tight')

def aggregated(records: list[SaveRecord]):
  plt.tight_layout()
  plt.figure(figsize=(12, 5))

  generations = len(records[0].history)
  ticks = range(1, generations + 1)

  for record in records:
    averages = np.array(record.select('avg'))
    stddevs = np.array(record.select('stddev'))

    plt.plot(ticks, averages, label=record.name)
    plt.fill_between(
      ticks,
      averages - stddevs,
      averages + stddevs,
      alpha=0.3
    )

  plt.ylabel('Fitness')
  plt.legend(
    loc='center left',
    bbox_to_anchor=(0.96, 0.5),
  )
  plt.savefig('resources/lab-1/figures/aggregated.png', bbox_inches='tight')


def main():
  records = [resources.read(name, model=SaveRecord) for name in resources.names('lab-1')]

  ensure_directory('resources/lab-1/figures')
  boxplot(records)
  plot(records)
  aggregated(records)

if __name__ == '__main__':
  main()
