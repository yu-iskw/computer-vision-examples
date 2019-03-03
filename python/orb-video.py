import cv2
import numpy as np

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

try:
    while True:
        # Capture frame-by-frame.
        ret, frame = cap.read()

        # Operate something on the frame come here.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Merge the images half-and-half
        frame[:, 0:(frame.shape[1] // 2), :] = hsv[:, 0:(frame.shape[1] // 2), :]

        # Display the converted frame.
        cv2.imshow('frame', frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
