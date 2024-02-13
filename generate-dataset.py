import mediapipe as mp
import cv2
import numpy as np
import sys

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

landmark_coords = np.zeros((21, 3))

with detect_hands.Hands(min_detection_confidence=.8, min_tracking_confidence=.5) as hands:

    # frameNum keeps track of the specific frame
    frameNum = 0
    while eye.isOpened():
        status, frame = eye.read()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        result = hands.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # prints the current frame
        print("Frame Number", frameNum)
        if result.multi_hand_landmarks:
            for num, hand in enumerate(result.multi_hand_landmarks):
                draw.draw_landmarks(img, hand, detect_hands.HAND_CONNECTIONS)

                # This loop gathers x, y, and z coordinates every 10 frames. Stores in landmark_coords array
                if frameNum % 10 == 0:
                    lm_num = 0
                    for coords in hand.landmark:
                        landmark_coords[lm_num] = [coords.x, coords.y, coords.x]
                        lm_num += 1

                    print(landmark_coords)

        cv2.imshow('Hand Fetish', img)

        frameNum += 1

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

eye.release()
cv2.destroyAllWindows()
