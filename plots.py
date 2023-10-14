# 3 graphs.
## 1. N best individuals. Y axis - fitness, X axis - generation.
## 2. Aggregated ( avg + std, avg, avg - std ) fitness. Y axis - fitness, X axis - generation.
## 3. Box plots of individuals' fitness. Y axis - fitness, X axis - generation.

import resources
from resources.models import SaveFile

def boxplot(data: list[SaveFile]):
  genotype_values = []
  for entry in data:
    genotype_values.append([genotype["vertpos"] for genotype in entry["genotypes"]])
  genotype_names = [entry["individual"] for entry in data]

  # Create a box plot
  plt.boxplot(genotype_values, labels=genotype_names, vert=True)
  plt.xlabel("Vertpos Values")
  plt.title("Box Plot of Vertpos Values")
  plt.show()

if __name__ == '__main__':
  import matplotlib.pyplot as plt
  data = resources.read("HoF-f9-mut-0.00", format="json", model=SaveFile)
  print(data)
  # print(data.name)

  # genotypes = [data]
  # boxplot(genotypes)
