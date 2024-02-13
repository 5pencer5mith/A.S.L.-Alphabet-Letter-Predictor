import pickle

# Interface for pickle package
def loadFromFile(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)
def saveToFile(filename, obj):
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
