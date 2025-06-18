import yaml
import cv2
import time
import argparse
import os
from datetime import datetime

from detector import load_model, load_class_names, run_detection
from alarm import trigger_alarm

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def log_violation(camera_id, frame_idx, violation_rate):
    ensure_dir("logs")
    with open("logs/alerts.log", "a") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{now}] Frame {frame_idx} — No helmet detected | Violation rate: {violation_rate:.2f}%\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='Video file path to test on')
    args = parser.parse_args()

    config = load_config()
    model = load_model(config['model_path'])
    class_names, helmet_class, no_helmet_class = load_class_names(config['class_file'])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("❌ Failed to open video.")
        return

    RESIZE_DIM = (416, 416)
    OUTPUT_DIR = "output"
    ensure_dir(OUTPUT_DIR)

    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    width, height = RESIZE_DIM
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    out = cv2.VideoWriter(f"{OUTPUT_DIR}/{video_name}_annotated.mp4", fourcc, fps, (width, height))

    frame_count = 0
    violation_count = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ End of video.")
            break

        frame_count += 1
        start_time = time.time()
        frame_resized = cv2.resize(frame, RESIZE_DIM)
        detections = run_detection(model, frame_resized, config['confidence_threshold'])

        # Trigger alarm if 'no_hats' detected
        violated = False
        for det in detections:
            if det['class'] == no_helmet_class:
                violation_count += 1
                violated = True
                violation_rate = (violation_count / frame_count) * 100
                trigger_alarm(0, config['alarm_sound_file'], config['alarm_cooldown_sec'])
                log_violation(0, frame_count, violation_rate)
                break

        # Draw results
        for det in detections:
            x1, y1, x2, y2 = map(int, det['box'])
            label = f"{det['class']} {det['conf']:.2f}"
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_resized, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        # === Calculate FPS ===
        end_time = time.time()
        fps_actual = 1 / (end_time - start_time + 1e-5)

        # Annotate frame with stats
        # stat_text = f"Frame: {frame_count}  Violations: {violation_count}  Rate: {violation_count / frame_count * 100:.2f}%"
        stat_text = f"Violations: {violation_count} FPS: {fps_actual:.2f}"
        cv2.putText(frame_resized, stat_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        cv2.imshow("Helmet Detection - Video", frame_resized)
        out.write(frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Interrupted by user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
