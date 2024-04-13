import util
import sys
import matplotlib.pyplot as plt
import numpy as np

# List of letters
alphabet = 'abcdefghijklmnopqrstuvwxyz'

# Which points to connect
connections = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [7, 8],
    [9, 10],
    [10, 11],
    [11, 12],
    [13, 14],
    [14, 15],
    [15, 16],
    [17, 18],
    [18, 19],
    [19, 20],
    [0, 17],
    [17, 13],
    [13, 9],
    [9, 5],
    [5, 2],
    [1, 5],
]

# Get command line args
name = sys.argv[1]
whichSampleToDraw = int(sys.argv[2])

# Save one image per letter
for letter in alphabet:
    # Get a sample for this letter
    filename = f"datasets/dataset-{letter}-{name}.pickle"
    data = util.loadFromFile(filename)
    data = np.array([data['samples'][whichSampleToDraw]])

    # Create a plot for this letter
    plt.cla()
    hand = data[0]
    lineSegments = [[hand[line[0]], hand[line[1]]] for line in connections]
    lineSegments = np.array(lineSegments).reshape(-1, 2)
    lineSegments = np.array([[point[0], 1-point[1]] for point in lineSegments])
    lineSegments = lineSegments.reshape(-1, 2, 2).swapaxes(1, 2).reshape(-1, 2)
    plt.plot(*lineSegments, c="k", marker="o")

    plt.savefig(f"dataset-images/{letter}.png")
