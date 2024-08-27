import mediapipe as mp
import cv2

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
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the image
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=15),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=10),
                )
                
                # Access and print landmarks of the palm (base of the fingers)
                # Here, we can access the landmarks directly if needed
                for id, landmark in enumerate(hand_landmarks.landmark):
                    # Get landmark position
                    h, w, c = image.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    
                    # Draw a circle at the palm's base
                    if id == 0:  # You might choose other landmarks depending on your requirement
                        cv2.circle(image, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        
        # Display the resulting frame
        cv2.imshow('Hand Palm Detection', image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
