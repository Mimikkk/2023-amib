from matplotlib import pyplot as plt
def main():
  # Data
  data = [
    {"avg": 0.2341493666274886},
    {"avg": 0.1},
    {"avg": 0.2},
    {"avg": 0.23},
    {"avg": 0.4},
  ]
  plt.clf()
  # Extract 'avg' values and their indices
  avg_values = [item["avg"] for item in data]
  indices = list(range(1, len(data) + 1))  # Generate indices for x-axis
  # Create a line plot
  plt.plot(indices, avg_values, marker='o', linestyle='-')

  # Add labels and title
  plt.title('Line Plot of "avg" Values')
  plt.xlabel('Data Point Index')
  plt.ylabel('Value')

  # Show the plot
  plt.grid(True)  # Add grid lines
  plt.show()


if __name__ == '__main__':
  main()