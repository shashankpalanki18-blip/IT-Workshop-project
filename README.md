# Real-Time ASL Sign Language Transcriber

A web-based computer vision application that recognizes static American Sign Language (ASL) fingerspelling in real-time. Built for our IT Workshop course, this project uses MediaPipe for hand landmark detection and a custom-trained Random Forest classifier to translate gestures into text directly through the browser.

## Features
*   **Real-Time Tracking:** Uses MediaPipe to instantly map 21 3D landmarks on the user's hand.
*   **Custom Dataset:** Trained on a custom-built dataset focusing on relative joint coordinates for robust, environment-independent recognition.
*   **Web Integration:** A Flask backend serves a lightweight HTML/JS frontend, allowing the app to run entirely in the browser.
*   **24-Letter Support:** Recognizes all static ASL letters (A-Z, excluding motion-based letters J and Z).

## Tech Stack
*   **Frontend:** HTML, CSS, JavaScript (Handles webcam capture and UI)
*   **Backend:** Python, Flask (Serves the web app and handles predictions)
*   **Machine Learning:** Scikit-Learn (Random Forest), Pandas (Data handling)
*   **Computer Vision:** OpenCV, MediaPipe (Hand landmark extraction)

---

## Local Setup Instructions (For Teammates)

Follow these steps to clone the repository and get the app running on your local machine.

### 1. Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone <YOUR-GITHUB-REPO-URL-HERE>
cd sign-language-app
