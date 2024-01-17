#Install these packages
import mediapipe as mp
import cv2


draw = mp.solutions.drawing_utils
detect_hands = mp.solutions.hands
#You may need to adjust the value in video capture (Usually it's 0 or 1) this creates a video object from your webcam
eye = cv2.VideoCapture(1)

with detect_hands.Hands(min_detection_confidence=.8, min_tracking_confidence=.5) as hands:
    while eye.isOpened():
        status, frame = eye.read()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        result = hands.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks:
            for num, hand in enumerate(result.multi_hand_landmarks):
                draw.draw_landmarks(img, hand, detect_hands.HAND_CONNECTIONS)

        cv2.imshow('Hand Fetish', img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

eye.release()
cv2.destroyAllWindows()
