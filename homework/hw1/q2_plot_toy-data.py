import matplotlib.pyplot as plt
import numpy as np

toy_data = np.load(f"data/toy-data.npz")
fields = "test_data", "training_data", "training_labels"
training_data = toy_data[fields[1]]
training_labels = toy_data[fields[2]]

# Plot the data points
plt.scatter(training_data[:, 0], training_data[:, 1], c=training_labels)

# Plot the decision boundary
x = np.linspace(-5, 5, 100)
w = np.array((-0.4528, -0.5190))
b = 0.1471
y = -(w[0] * x + b) / w[1]
plt.plot(x, y, 'k')

# Plot the margins
y_margin1 = -(w[0] * x + b - 1) / w[1]
y_margin2 = -(w[0] * x + b + 1) / w[1]
plt.plot(x, y_margin1, 'k', c='red')
plt.plot(x, y_margin2, 'k', c='red')

plt.savefig("q1_plot_toy-data.png")
plt.show()