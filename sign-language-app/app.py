import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import time

# 1. Download the Hand Landmarker model automatically if missing
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading MediaPipe AI model (this only happens once)...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete.")

# 2. Setup the modern Tasks API for Hand Landmarking
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO, # Video mode requires timestamps
    num_hands=2)

detector = vision.HandLandmarker.create_from_options(options)

# 3. Start the webcam
cap = cv2.VideoCapture(0)
print("Webcam opened. Press 'q' in the video window to close it.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    # Flip the frame for a mirror-like view
    frame = cv2.flip(frame, 1)
    
    # MediaPipe Tasks API requires a specific mp.Image object format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Process the frame with a timestamp (in milliseconds)
    frame_timestamp_ms = int(time.time() * 1000)
    detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
    
    # 4. Manually draw green dots on the joints using pure OpenCV
    if detection_result.hand_landmarks:
        for hand in detection_result.hand_landmarks:
            for landmark in hand:
                # Convert normalized coordinates (0.0 to 1.0) to actual pixel locations
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Show the video feed
    cv2.imshow('MediaPipe Hand Tracking Test', frame)
    
    # Listen for the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()