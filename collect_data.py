import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time
import csv
import urllib.request

# 1. Download the Hand Landmarker model automatically if missing
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading MediaPipe AI model (this only happens once)...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete.")

# 2. Setup MediaPipe
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1) # Limit to 1 hand for ASL
detector = vision.HandLandmarker.create_from_options(options)

# 2. Prepare the CSV file in the data folder
csv_file = 'data/dataset.csv'
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        # Create headers: label, x0...x20, y0...y20
        headers = ['label'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)]
        csv.writer(f).writerow(headers)

# 3. Start Data Collection
cap = cv2.VideoCapture(0)
print("INSTRUCTIONS:")
print("- Hold up a sign for a letter (e.g., 'A').")
print("- Press that letter key on your keyboard to save a sample.")
print("- Move your hand slightly and press it again to get varied data.")
print("- Press '1' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.flip(frame, 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = detector.detect_for_video(mp_image, int(time.time() * 1000))
    
    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        
        # Draw the dots
        for landmark in hand:
            cv2.circle(frame, (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])), 5, (0, 255, 0), -1)
            
        # Listen for keyboard presses
        key = cv2.waitKey(1) & 0xFF
        if ord('a') <= key <= ord('z'):
            label = chr(key).upper()
            wrist_x, wrist_y = hand[0].x, hand[0].y # Reference point
            
            # Calculate coordinates relative to the wrist
            row = [label]
            row.extend([lm.x - wrist_x for lm in hand]) # X coordinates
            row.extend([lm.y - wrist_y for lm in hand]) # Y coordinates
            
            with open(csv_file, mode='a', newline='') as f:
                csv.writer(f).writerow(row)
            print(f"Captured: {label}")
            
        elif key == ord('1'):
            break
    else:
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break

    cv2.imshow('Data Collector', frame)

cap.release()
cv2.destroyAllWindows()