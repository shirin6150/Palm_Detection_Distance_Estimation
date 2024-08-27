import cv2
import numpy as np
import mediapipe as mp

# Load calibration data from npz file
with np.load('MultiMatrix_2.npz') as data:
    camera_matrix = data['camMatrix']
    dist_coeffs = data['distCoef']

# Initialize MediaPipe Hands and Drawing Utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Setup MediaPipe Hands with specific detection and tracking confidences
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Undistort the image using the calibration data
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, camera_matrix)

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image to find hands
        results = hands.process(image)

        # Convert the RGB image back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw hand landmarks on the image
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2),
                )

                # Calculate distance between two landmarks (e.g., wrist and tip of the middle finger)
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                # Convert the normalized coordinates to pixel coordinates
                h, w, c = image.shape
                wrist_px = (int(wrist.x * w), int(wrist.y * h))
                middle_finger_tip_px = (int(middle_finger_tip.x * w), int(middle_finger_tip.y * h))

                # Calculate the pixel distance between these landmarks
                pixel_distance = np.linalg.norm(np.array(wrist_px) - np.array(middle_finger_tip_px))

                # Assuming the actual distance between wrist and middle finger tip is around 17 cm
                real_distance = 20  # in centimeters

                # Focal length in pixels, calculated during calibration
                focal_length = camera_matrix[0, 0]  # Focal length is usually at [0, 0]

                # Distance estimation using triangular similarity
                estimated_distance = (focal_length * real_distance) / pixel_distance

                # Determine whether the hand is left or right
                hand_label = handedness.classification[0].label

                # Display the estimated distance with a label for each hand
                if hand_label == 'Right':
                    cv2.putText(image, f"Left Hand Distance: {estimated_distance:.2f} cm", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif hand_label == 'Left':
                    cv2.putText(image, f"Right Hand Distance: {estimated_distance:.2f} cm", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('Hand Palm Distance Estimation', image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()





