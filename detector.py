from ultralytics import YOLO
import yaml

def load_model(model_path):
    return YOLO(model_path)

def load_class_names(path):
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names'], data['helmet_class'], data['no_helmet_class']

def run_detection(model, frame, confidence):
    results = model.predict(source=frame, conf=confidence, stream=False, verbose=False)[0]
    detections = []
    for i, box in enumerate(results.boxes.data):
        x1, y1, x2, y2, conf, cls = box.tolist()
        class_id = int(cls)
        class_name = results.names[class_id]
        if class_name == "ignore":
            continue  # Skip class 0 (was 'dog')
        detections.append({
            'id': i,
            'box': [x1, y1, x2, y2],
            'conf': conf,
            'class': class_name
        })
    return detections
