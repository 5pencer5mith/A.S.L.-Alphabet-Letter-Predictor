import mediapipe as mp
import cv2
import numpy as np
import sys
import util
import time

# Get command line arguments
currentLetter = sys.argv[1]
datasetVersion = sys.argv[2]
outputFilename = "dataset-%s-%s.pickle" % (currentLetter, datasetVersion)

# Collection place for dataset
dataset = {
    "letter": currentLetter,
    "samples": []
}

# Setting up mediapipe
draw = mp.solutions.drawing_utils
detect_hands = mp.solutions.hands

# Setting up Video Capture
eye = cv2.VideoCapture(1)
if eye is None or not eye.isOpened():
    eye = cv2.VideoCapture(0)

# Scan the camera, and detect hands
with detect_hands.Hands(min_detection_confidence=.8, min_tracking_confidence=.5) as hands:
    # frameNum keeps track of the specific frame
    frameNum = 0
    # Start looking at the camera
    while eye.isOpened():
        # Get a frame
        status, frame = eye.read()
        # Convert image from BGR to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Get hand information from image
        img.flags.writeable = False
        result = hands.process(img)
        img.flags.writeable = True
        # Convert image back to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Do stuff if hands were found in the image
        if result.multi_hand_landmarks:
            # Do something with each hand that was found
            for num, hand in enumerate(result.multi_hand_landmarks):
                # Draw hand wireframe on the image
                draw.draw_landmarks(img, hand, detect_hands.HAND_CONNECTIONS)
                # This loop gathers x, y, and z coordinates every 10 frames. Stores in landmark_coords array
                if frameNum % 10 == 0: # only look at every 10 frames
                    # Process landmarks
                    coords = util.processLandmarks(hand.landmark)
                    # Add to dataset
                    dataset['samples'].append(coords)
                    # Alert user how many samples have been collected so far
                    print("Samples collected: %d" % len(dataset['samples']))
        # Display the image with the wireframes
        cv2.imshow('Hand Fetish', img)
        # Increment frame count
        frameNum += 1
        # Quit if the q key is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
# Close modules
eye.release()
cv2.destroyAllWindows()
# Save dataset to file
util.saveToFile(outputFilename, dataset)
