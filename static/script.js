import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const predictionText = document.getElementById("prediction");

let handLandmarker = undefined;
let lastVideoTime = -1;

// 1. Initialize the MediaPipe Hand Landmarker
async function createHandLandmarker() {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 1
    });
    startWebcam();
}

// 2. Turn on the webcam
function startWebcam() {
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    });
}

// 3. Process the video frames continuously
async function predictWebcam() {
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;

    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        
        // Detect hands
        const results = handLandmarker.detectForVideo(video, startTimeMs);
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

        if (results.landmarks && results.landmarks.length > 0) {
            const hand = results.landmarks[0];
            
            // Draw green dots on joints
            for (const lm of hand) {
                canvasCtx.fillStyle = "#00FF00";
                canvasCtx.beginPath();
                canvasCtx.arc(lm.x * canvasElement.width, lm.y * canvasElement.height, 5, 0, 2 * Math.PI);
                canvasCtx.fill();
            }

            // Extract coordinates relative to the wrist (index 0)
            const wristX = hand[0].x;
            const wristY = hand[0].y;
            let formattedLandmarks = [];
            
            // Add X coordinates
            for (const lm of hand) formattedLandmarks.push(lm.x - wristX);
            // Add Y coordinates
            for (const lm of hand) formattedLandmarks.push(lm.y - wristY);

            // Send to Flask backend
            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ landmarks: formattedLandmarks })
            })
            .then(response => response.json())
            .then(data => {
                if (data.letter) predictionText.innerText = data.letter;
            });
        }
    }
    window.requestAnimationFrame(predictWebcam);
}

// Start the sequence
createHandLandmarker();