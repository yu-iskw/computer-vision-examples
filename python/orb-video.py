import cv2
import copy

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

try:
    while True:
        # Capture frame-by-frame.
        ret, frame = cap.read()

        # Operate something on the frame come here.
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Extract the key points with ORB.
        orb = cv2.ORB_create(200, 2.0)
        keypoints, descriptor = orb.detectAndCompute(frame_gray, None)

        # Show the frame with key points.
        keyp_without_size = copy.copy(frame)
        keyp_with_size = copy.copy(frame)
        # Draw the keypoints without size or orientation on one copy of the training image
        cv2.drawKeypoints(frame, keypoints, keyp_without_size, color=(0, 255, 0))
        # Draw the keypoints with size and orientation on the other copy of the training image
        cv2.drawKeypoints(frame, keypoints, keyp_with_size,
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Display the converted frame.
        cv2.imshow('frame', keyp_without_size)
        # cv2.imshow('frame', keyp_with_size)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
