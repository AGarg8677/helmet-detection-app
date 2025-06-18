import cv2
from ultralytics import YOLO

# === CONFIGURATION ===
VIDEO_PATH = "clip6.mp4"           # <-- change to your actual video path
FRAME_TIMESTAMP_SEC = 3                 # Get frame at 3 seconds
RESIZE_DIM = (640, 640)                 # Resize to square
MODEL_PATH = "weights/helmet_model.pt" # Your YOLOv8 model
CONFIDENCE = 0.25                       # Lowered confidence for testing

# === LOAD VIDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_index = int(FRAME_TIMESTAMP_SEC * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

ret, frame = cap.read()
if not ret:
    print("Failed to read frame from video.")
    exit()

# === RESIZE FRAME ===
resized = cv2.resize(frame, RESIZE_DIM)

# === RUN YOLOv8 PREDICTION ===
model = YOLO(MODEL_PATH)
results = model.predict(resized, conf=CONFIDENCE, show=False)

# === DRAW DETECTIONS ===
annotated = results[0].plot()

# === SAVE & SHOW ===
cv2.imwrite("annotated_debug_frame.jpg", annotated)
cv2.imshow("Detections", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
