import util
import sys
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
fileName = sys.argv[1]
data = util.loadFromFile(fileName)
data = np.array(data['samples'])

# Reshape the data
size = len(data) * len(data[0])
data = data.reshape((size, 2))

# Plot the data
plt.scatter(data[:,0], 1-data[:,1])
plt.show()
