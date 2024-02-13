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
    # Used later to temporarily hold hand wireframe coords
    landmark_coords = np.zeros((21, 3))
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
                    # Loop through landmarks, and store them in landmark_coords
                    lm_num = 0
                    for coords in hand.landmark:
                        landmark_coords[lm_num] = [coords.x, coords.y, coords.z]
                        lm_num += 1
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
