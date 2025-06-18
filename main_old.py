# import yaml
# import cv2
# import time
# import argparse
# # from ultralytics import YOLO

# from camera_stream import get_camera_streams
# from detector import load_model, load_class_names, run_detection
# from utils import match_helmets_to_people
# from alarm import trigger_alarm

# def load_config(path='config.yaml'):
#     with open(path, 'r') as f:
#         return yaml.safe_load(f)

# # For sample video files
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--video', type=str, default=None, help='Optional video file path')
#     args = parser.parse_args()

#     config = load_config()
#     model = load_model(config['model_path'])
#     class_names, helmet_class, person_class = load_class_names(config['class_file'])

#     # Video test mode
#     if args.video:
#         print(f"[TEST MODE] Using video file: {args.video}")
#         caps = [cv2.VideoCapture(args.video)]
#     else:
#         from camera_stream import get_camera_streams
#         caps = get_camera_streams(config['camera_feeds'])

#     while True:
#         for cam_id, cap in enumerate(caps):
#             ret, frame = cap.read()
#             if not ret:
#                 print(f"End of stream or failed read from camera {cam_id}")
#                 continue

#             detections = run_detection(model, frame, config['confidence_threshold'])

#             if match_helmets_to_people(detections, helmet_class, person_class):
#                 trigger_alarm(cam_id, config['alarm_sound_file'], config['alarm_cooldown_sec'])

#             cv2.imshow(f'Camera {cam_id}', frame)
#             if cv2.waitKey(1) == ord('q'):
#                 break

#         if args.video and not caps[0].isOpened():
#             break

#     for cap in caps:
#         cap.release()
#     cv2.destroyAllWindows()


# for IP camera streams
# def main():
#     config = load_config()
#     model = load_model(config['model_path'])
#     class_names, helmet_class, person_class = load_class_names(config['class_file'])

#     caps = get_camera_streams(config['camera_feeds'])

#     RESIZE_DIM = (416, 416)  # Optimized for better FPS on CPU

#     while True:
#         for cam_id, cap in enumerate(caps):
#             ret, frame = cap.read()
#             if not ret or frame is None:
#                 print(f"Failed to read from camera {cam_id}")
#                 continue

#             # Resize the frame before inference
#             frame_resized = cv2.resize(frame, RESIZE_DIM)

#             # Run detection on resized frame
#             detections = run_detection(model, frame_resized, config['confidence_threshold'])

#             # Trigger alarm if violation
#             if match_helmets_to_people(detections, helmet_class, person_class):
#                 trigger_alarm(cam_id, config['alarm_sound_file'], config['alarm_cooldown_sec'])

#             # Draw detection results on resized frame
#             for det in detections:
#                 x1, y1, x2, y2 = map(int, det['box'])
#                 label = f"{det['class']} {det['conf']:.2f}"
#                 cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame_resized, label, (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#             # Display the resized annotated frame
#             cv2.imshow(f'Camera {cam_id}', frame_resized)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 print("Exiting by user command.")
#                 break

#         time.sleep(0.1)  # Controls loop speed (can tune for performance)

#     for cap in caps:
#         cap.release()
#     cv2.destroyAllWindows()


# if __name__ == '__main__':
#     main()
import yaml
import cv2
import time
import argparse
import os
from datetime import datetime

from camera_stream import get_camera_streams
from detector import load_model, load_class_names, run_detection
from utils import match_helmets_to_people
from alarm import trigger_alarm

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def log_violation(camera_id):
    ensure_dir("logs")
    with open("logs/alerts.log", "a") as f:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{now}] ALERT: No helmet detected on Camera {camera_id}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default=None, help='Optional video file path')
    args = parser.parse_args()

    config = load_config()
    model = load_model(config['model_path'])
    class_names, helmet_class, person_class = load_class_names(config['class_file'])

    RESIZE_DIM = (416, 416) # (640, 640)
    OUTPUT_DIR = "output"
    ensure_dir(OUTPUT_DIR)

    if args.video:
        print(f"[TEST MODE] Using video file: {args.video}")
        caps = [cv2.VideoCapture(args.video)]
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        fps = caps[0].get(cv2.CAP_PROP_FPS) or 10  # fallback fps
        width, height = RESIZE_DIM
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"{OUTPUT_DIR}/{video_name}_annotated.mp4", fourcc, fps, (width, height))
    else:
        caps = get_camera_streams(config['camera_feeds'])
        out = None

    while True:
        for cam_id, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                print(f"End of stream or failed read from camera {cam_id}")
                continue

            frame_resized = cv2.resize(frame, RESIZE_DIM)
            detections = run_detection(model, frame_resized, config['confidence_threshold'])

            violation = match_helmets_to_people(detections, helmet_class, person_class)
            if violation:
                trigger_alarm(cam_id, config['alarm_sound_file'], config['alarm_cooldown_sec'])
                log_violation(cam_id)

            for det in detections:
                x1, y1, x2, y2 = map(int, det['box'])
                label = f"{det['class']} {det['conf']:.2f}"
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_resized, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Show output
            cv2.imshow(f'Camera {cam_id}', frame_resized)

            # Save frame to output video
            if args.video and out:
                out.write(frame_resized)

            # Increase FPS playback (delay in ms)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 Interrupted by user.")
                break

        if args.video and not caps[0].isOpened():
            break

    for cap in caps:
        cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
