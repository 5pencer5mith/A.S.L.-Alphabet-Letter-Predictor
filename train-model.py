import numpy as np
import util
import glob

# Import all datasets
datasets = []
for filepath in glob.iglob('datasets/*.pickle'):
    datasets.append(util.loadFromFile(filepath))

# Combine datasets
X = np.array([sample for dataset in datasets for sample in dataset['samples']])
Y = np.array([letter for dataset in datasets for letter in [dataset['letter'] for sample in dataset['samples']]])

# Reshape each sample input
X = X.reshape((len(X), len(X[0]) * 2))

# Transform letters into classifications
letters = 'abcdefghijklmnopqrstuvwxyz'
classDict = {letter:np.zeros((26)) for letter in letters}
for letter, index in zip(letters, range(26)):
    classDict[letter][index] = 1.
Y = np.array([classDict[letter] for letter in Y])

# Split into training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20)

# Train model
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=10000000)
mlp.fit(X_train, Y_train)

# Test model
# Generate predictions
predictions = mlp.predict(X_test)
# Use max to choose a letter
predictions = [util.indexOfMax(y) for y in predictions]
Y_test = [util.indexOfMax(y) for y in Y_test]
predictions = [letters[y] for y in predictions]
Y_test = [letters[y] for y in Y_test]
# Calculate how many predictions were correct
comparisons = [1 if y == p else 0 for y, p in zip(Y_test, predictions)]
# Print accuracy
print("Accuracy: %.3f" % np.mean(comparisons))

# Save model
util.saveToFile('model.pickle', mlp)
