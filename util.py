import pickle

# Interface for pickle package
def loadFromFile(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)
def saveToFile(filename, obj):
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Process landmarks from mediapipe
def processLandmarks(landmarks):
    landmark_coords = []
    # Loop through landmarks, and store them in landmark_coords
    for coords in landmarks:
        landmark_coords.append([coords.x, coords.y, coords.z])
    # Find bounding box of hand
    minx = landmark_coords[0][0]
    maxx = landmark_coords[0][0]
    miny = landmark_coords[0][1]
    maxy = landmark_coords[0][1]
    for coords in landmark_coords:
        if coords[0] < minx:
            minx = coords[0]
        if coords[0] > maxx:
            maxx = coords[0]
        if coords[1] < miny:
            miny = coords[1]
        if coords[1] > maxy:
            maxy = coords[1]
    width = maxx - minx
    height = maxy - miny
    # Create new list of scaled landmarks
    convertedCoords = []
    for coords in landmark_coords:
        convertedCoords.append([
            (coords[0] - minx) / width,
            (coords[1] - miny) / height
        ])
    # Return converted coords
    return convertedCoords
# Utilities to help with the model
def indexOfMax(arr):
    bestSoFar = arr[0]
    indexOfBest = 0
    for i in range(1, len(arr)):
        if arr[i] > bestSoFar:
            bestSoFar = arr[i]
            indexOfBest = i
    return indexOfBest
